# NVIDIA Jetson Thor X Instruction Simulator — Design Document

## 1. Overview

This document describes the design of an instruction-level simulator for the **NVIDIA Jetson Thor X** SoC. Given an arbitrary instruction sequence (CPU or GPU), the simulator:

1. Parses and decodes each instruction.
2. Simulates execution, tracking FLOPs and memory-traffic events.
3. Produces an **offline Roofline model** — a per-kernel chart that maps attainable performance against arithmetic intensity relative to Thor X hardware ceilings.

### 1.1 Glossary

| Term | Meaning |
|------|---------|
| **Roofline model** | Log–log plot of attainable GFLOPs/s vs. arithmetic intensity (FLOPs/byte). Two ceilings: peak compute and peak bandwidth. |
| **Arithmetic Intensity (AI)** | FLOPs executed ÷ bytes transferred to/from memory. |
| **Ridge point** | AI where peak-compute ceiling meets peak-bandwidth diagonal. |
| **Offline analysis** | Simulation driven from a static instruction trace; no real hardware required. |

---

## 2. Target Hardware: NVIDIA Jetson Thor X

> Note: Ceiling values in this section are provisional design inputs and MUST be validated against official Thor X documentation before implementation sign-off.

### 2.1 Key Specifications

| Subsystem | Specification |
|-----------|--------------|
| CPU | 12× Arm Cortex-A78AE @ ~3.2 GHz (Hercules-AE, ARMv8.2-A) |
| GPU | NVIDIA Ampere — 2 048 CUDA cores |
| DLA | 2× Deep Learning Accelerator v3.0 |
| Memory | LPDDR5 — 256-bit bus |
| Peak CPU FP64 | ~76.8 GFLOP/s (provisional) |
| Peak GPU FP32 | ~16 TFLOP/s |
| Peak GPU FP16 | ~32 TFLOP/s |
| Peak mem bandwidth (CPU→DRAM) | ~68 GB/s |
| Peak mem bandwidth (GPU→DRAM) | ~204 GB/s |
| L1 cache (CPU, per core) | 64 KB |
| L2 cache (CPU, shared) | 4 MB |
| GPU L2 | 4 MB |

### 2.2 Supported Instruction Domains

The simulator covers three instruction domains:

| Domain | ISA | Notes |
|--------|-----|-------|
| `cpu` | ARMv8.2-A (AArch64) | NEON SIMD in scope; SVE support is out of scope until hardware confirmation |
| `gpu` | NVIDIA PTX ISA (virtual) | Mapped to Ampere micro-ops |
| `dla` | NVDLA layer descriptor | Convolution / activation primitives |

---

## 3. Functional Requirements

| ID | Requirement |
|----|------------|
| FR-01 | Accept instruction traces in JSON, assembly text, PTX, or DLA-JSON format. |
| FR-02 | Decode every instruction into: mnemonic, operands, FLOP count, memory access size. |
| FR-03 | Accumulate per-kernel metrics: total FLOPs, DRAM bytes read/written, and on-chip bytes read/written. |
| FR-04 | Compute arithmetic intensity for each kernel. |
| FR-05 | Emit a Roofline model report (JSON + SVG) per hardware domain. |
| FR-06 | Report whether each kernel is compute-bound or memory-bandwidth-bound. |
| FR-07 | Support multiple kernels in one trace (delimited by kernel markers). |
| FR-08 | CLI and Python API interfaces. |

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          thor-sim (top-level)                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Trace Input │───▶│    Parser    │───▶│  Instruction Stream  │  │
│  │ (JSON/ASM/   │    │  (per-domain)│    │  (decoded IR)        │  │
│  │ PTX/DLA-JSON)│    └──────────────┘    └──────────┬───────────┘  │
│  └──────────────┘                                   │              │
│                                                     ▼              │
│                                         ┌──────────────────────┐   │
│                                         │  Execution Engine    │   │
│                                         │  - FLOP counter      │   │
│                                         │  - Mem traffic model │   │
│                                         │  - Cache simulator   │   │
│                                         └──────────┬───────────┘   │
│                                                    │               │
│                                                    ▼               │
│                                         ┌──────────────────────┐   │
│                                         │  Performance Counter  │   │
│                                         │  Store (per kernel)  │   │
│                                         └──────────┬───────────┘   │
│                                                    │               │
│                                                    ▼               │
│                                         ┌──────────────────────┐   │
│                                         │  Roofline Analyzer   │   │
│                                         │  - AI computation    │   │
│                                         │  - Ceiling selection │   │
│                                         │  - Bound detection   │   │
│                                         └──────────┬───────────┘   │
│                                                    │               │
│                                                    ▼               │
│                                         ┌──────────────────────┐   │
│                                         │  Report Generator    │   │
│                                         │  (JSON + SVG + text) │   │
│                                         └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.1 Component Responsibilities

#### 4.1.1 Parser

- **Input**: raw trace file (path or string).
- **Output**: list of `DecodedInstruction` objects grouped by kernel.
- Sub-parsers: `AArch64Parser`, `PTXParser`, `DLAParser`.
- Each sub-parser implements `parse(source: str) -> List[DecodedInstruction]`.

#### 4.1.2 Execution Engine

Simulates in-order execution of the decoded stream. For each instruction it:

- Adds the instruction's FLOP contribution to `kernel.total_flops`.
- Resolves memory operands through the **Cache Simulator** to determine whether data hits cache or goes to DRAM, adding the corresponding bytes to `kernel.bytes_dram_*` or `kernel.bytes_onchip_*`.

The cache simulator uses a simplified LRU set-associative model matching Thor X cache parameters.

#### 4.1.3 Performance Counter Store

```
KernelMetrics {
    name:           str
    domain:         Domain          # CPU | GPU | DLA
    instruction_count: int
    total_flops:    float           # summed FLOP weights
    bytes_dram_read: int            # DRAM reads (post cache)
    bytes_dram_written: int         # DRAM writes (post cache)
    bytes_onchip_read: int          # cache/shared/local reads
    bytes_onchip_written: int       # cache/shared/local writes
    arithmetic_intensity: float     # computed = total_flops / (bytes_dram_read + bytes_dram_written)
}
```

#### 4.1.4 Roofline Analyzer

For each `KernelMetrics`:

1. Select hardware domain ceilings (see §5.1).
2. Compute **attainable performance**:
   ```
   P_attainable = min(P_peak_compute,  AI × BW_peak)
   ```
   where `AI = total_flops / bytes_dram_total`.
   If `bytes_dram_total == 0`, define `AI = +inf` and classify as compute-bound.
3. Determine binding constraint:
   - `AI < ridge_point`  →  **memory-bandwidth bound**
   - `AI >= ridge_point` →  **compute bound**
4. Optionally project theoretical speedup to ridge point.

#### 4.1.5 Report Generator

Produces:

- `report.json` — full metrics + Roofline data per kernel.
- `roofline.svg` — log–log Roofline chart with kernel scatter points.
- `summary.txt` — human-readable table.

---

## 5. Roofline Model Specification

### 5.1 Hardware Ceilings per Domain

#### CPU Domain (Cortex-A78AE)

| Ceiling | Value | Formula |
|---------|-------|---------|
| Peak FP64 scalar | 76.8 GFLOP/s | 2 FLOP/cycle × 12 cores × 3.2 GHz |
| Peak FP32 NEON (128-bit) | 307.2 GFLOP/s | 8 FLOP/cycle (4-lane FP32 FMA) × 12 cores × 3.2 GHz |
| Peak FP16 NEON | 614.4 GFLOP/s | 16 FLOP/cycle (8-lane FP16 FMA) × 12 cores × 3.2 GHz |
| Peak DRAM BW | 68 GB/s | LPDDR5 measured |
| Peak L2 BW | ~820 GB/s | estimated |

#### GPU Domain (Ampere)

| Ceiling | Value |
|---------|-------|
| Peak FP32 | 16 TFLOP/s |
| Peak FP16 (Tensor Core) | 32 TFLOP/s |
| Peak INT8 (Tensor Core) | 64 TOPS |
| Peak DRAM BW | 204 GB/s |
| Peak L2 BW | ~2 TB/s |

### 5.2 Ridge Points

```
ridge_CPU_FP32  = P_peak_CPU_FP32  / BW_CPU_DRAM  ≈ 307.2 / 68   ≈ 4.52 FLOP/byte
ridge_GPU_FP32  = P_peak_GPU_FP32  / BW_GPU_DRAM  ≈ 16 000 / 204 ≈ 78.4 FLOP/byte
ridge_GPU_FP16  = P_peak_GPU_FP16  / BW_GPU_DRAM  ≈ 32 000 / 204 ≈ 156.9 FLOP/byte
```

For INT8/TOPS ceilings, the same ridge formula applies but with `OPS` (not `FLOPs`) in both numerator and throughput axis units.

### 5.3 Roofline Chart Layout

```
 GFLOPs/s
 (log)
  │  Peak Compute ─────────────────────────────────── P_peak
  │                                      ╱
  │                       ╱ bandwidth  ╱
  │              ╱  line (slope=BW)   ╱
  │       ╱                         ╱  ridge point
  │  ╱                             •
  └─────────────────────────────────────────────────▶ AI (FLOP/byte, log)
```

Each kernel is plotted as a labelled dot. Dots to the left of the ridge are shaded red (memory bound); to the right are shaded green (compute bound).

---

## 6. Instruction FLOP / Byte Weight Table

### 6.1 AArch64 Weights (examples)

| Mnemonic | FLOPs | DRAM bytes | On-chip bytes |
|----------|-------|------------|---------------|
| `fadd`, `fsub`, `fmul` | 1 | 0 | 0 |
| `fmadd`, `fmla` | 2 | 0 | 0 |
| `fdiv` | 1 | 0 | 0 |
| `ldr` (64-bit) | 0 | 8 | 0 |
| `str` (64-bit) | 0 | 8 | 0 |
| `ld1` (128-bit NEON) | 0 | 16 | 0 |
| `fmla.4s` (NEON 4×FP32) | 8 (4×FMA) | 0 | 0 |
| `fcvt` | 1 | 0 | 0 |

### 6.2 PTX Weights (examples)

| PTX instruction | FLOPs | DRAM bytes | On-chip bytes |
|-----------------|-------|------------|---------------|
| `add.f32`, `mul.f32` | 1 | 0 | 0 |
| `fma.f32` | 2 | 0 | 0 |
| `ld.global.f32` | 0 | 4 | 0 |
| `ld.shared.f32` | 0 | 0 | 4 |
| `st.global.f32` | 0 | 4 | 0 |
| `dp4a` (INT8 dot product) | 8 | 0 | 0 |
| `mma.m16n8k16` (FP16 Tensor) | 512 | 0 | 0 |

### 6.3 Extension Points

The weight tables are stored in `config/weights_cpu.yaml` and `config/weights_gpu.yaml`, enabling users to override or extend weights without code changes.
For AI used in Roofline classification, only DRAM bytes are included in the denominator; on-chip bytes are reported separately for diagnostics.

---

## 7. Input Trace Format

### 7.1 JSON Trace (canonical)

```json
{
  "domain": "gpu",
  "kernels": [
    {
      "name": "matmul_fp32",
      "instructions": [
        { "op": "ld.global.f32", "dst": "r0", "src": "[base+0]" },
        { "op": "fma.f32",       "dst": "r2", "src": ["r0","r1","r2"] },
        { "op": "st.global.f32", "dst": "[out+0]", "src": "r2" }
      ]
    }
  ]
}
```

### 7.2 Assembly Text Trace

Plain assembly text with optional kernel delimiters:

```asm
; kernel: conv2d
fmla    v0.4s,  v1.4s,  v2.4s
ldr     q3,     [x0], #16
fmla    v0.4s,  v3.4s,  v4.4s
str     q0,     [x1], #16
; end_kernel
```

### 7.3 PTX Trace

Standard NVIDIA PTX `.ptx` files. The parser extracts `.func` bodies as kernels.

### 7.4 DLA Trace (JSON)

DLA input uses a JSON schema instead of raw descriptors so that P0/P1 can run without undocumented firmware formats.

```json
{
  "domain": "dla",
  "kernels": [
    {
      "name": "conv2d_block0",
      "layers": [
        {
          "op": "conv2d",
          "precision": "int8",
          "input_shape": [1, 64, 56, 56],
          "weight_shape": [64, 64, 3, 3],
          "output_shape": [1, 64, 56, 56],
          "estimated_ops": 231211008,
          "estimated_bytes_dram": 6422528
        }
      ]
    }
  ]
}
```

Required layer fields: `op`, `precision`, `input_shape`, `output_shape`, `estimated_ops`, `estimated_bytes_dram`.

---

## 8. Output Report Format

### 8.1 JSON Report

```json
{
  "hardware": "Jetson ThorX / Ampere GPU",
  "domain": "gpu",
  "roofline_ceilings": {
    "peak_compute_gflops": 16000,
    "peak_bandwidth_gb_per_s": 204,
    "ridge_point_flop_per_byte": 78.4
  },
  "kernels": [
    {
      "name": "matmul_fp32",
      "instruction_count": 1024,
      "total_flops": 2097152,
      "bytes_dram_read": 65536,
      "bytes_dram_written": 65536,
      "bytes_onchip_read": 0,
      "bytes_onchip_written": 0,
      "arithmetic_intensity": 16.0,
      "attainable_gflops": 3264.0,
      "bound": "memory",
      "utilization_pct": 20.4,
      "edge_case": null
    }
  ]
}
```

### 8.2 SVG Chart

Generated by the built-in SVG renderer (no external plotting dependency required). Chart dimensions: 900×600 px, log–log axes.

---

## 9. Project Structure

```
nv-thorX/
├── DESIGN.md                   # this document
├── README.md
├── pyproject.toml
├── config/
│   ├── hardware_thorx.yaml     # Thor X hardware ceilings
│   ├── weights_cpu.yaml        # AArch64 FLOP/byte weights
│   └── weights_gpu.yaml        # PTX FLOP/byte weights
├── thor_sim/
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point (argparse)
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── base.py             # BaseParser ABC
│   │   ├── aarch64.py          # AArch64Parser
│   │   ├── ptx.py              # PTXParser
│   │   └── dla.py              # DLAParser
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── execution.py        # ExecutionEngine
│   │   └── cache.py            # LRU cache simulator
│   ├── roofline/
│   │   ├── __init__.py
│   │   ├── analyzer.py         # RooflineAnalyzer
│   │   └── hardware.py         # HardwareSpec (loaded from YAML)
│   └── report/
│       ├── __init__.py
│       ├── json_report.py
│       ├── svg_chart.py
│       └── text_summary.py
└── tests/
    ├── test_parser.py
    ├── test_engine.py
    ├── test_roofline.py
    └── fixtures/
        ├── matmul.json
        ├── conv2d.asm
        └── transformer.ptx
```

---

## 10. API

### 10.1 Python API

```python
from thor_sim import Simulator

sim = Simulator(domain="gpu")
report = sim.run("path/to/trace.ptx")

for kernel in report.kernels:
    print(f"{kernel.name}: AI={kernel.arithmetic_intensity:.1f} FLOP/byte  bound={kernel.bound}")

report.save_json("out/report.json")
report.save_svg("out/roofline.svg")
```

### 10.2 CLI

```bash
# Analyze a PTX trace
thor-sim analyze --domain gpu --input trace.ptx --out-dir ./results

# Analyze ARM assembly
thor-sim analyze --domain cpu --input kernel.asm --format asm

# Print text summary only
thor-sim analyze --domain gpu --input trace.json --no-svg
```

---

## 11. Implementation Plan

| Phase | Deliverable | Priority |
|-------|------------|---------|
| P0 | Hardware spec YAML + weight tables | Critical |
| P0 | `BaseParser` + JSON trace parser | Critical |
| P0 | `ExecutionEngine` (no cache) + counter store | Critical |
| P0 | `RooflineAnalyzer` + JSON report | Critical |
| P0 | Doc consistency checks (units/formula/ridge/schema required fields) | Critical |
| P1 | `AArch64Parser` (subset: NEON + scalar FP) | High |
| P1 | `PTXParser` (scalar + Tensor Core ops) | High |
| P1 | SVG chart renderer | High |
| P2 | LRU cache simulator | Medium |
| P2 | `DLAParser` | Medium |
| P2 | CLI packaging (`pyproject.toml`) | Medium |
| P3 | Multi-level roofline (L1/L2/DRAM ceilings) | Low |
| P3 | HTML interactive report (Plotly) | Low |

---

## 12. Open Questions

1. **Trace granularity**: Should the simulator support warp-level PTX execution (modeling occupancy) or purely instruction-count-based analysis?
2. **Tensor Core weighting**: The `mma` instruction throughput depends on operand precision and matrix shape. Should precision variants be enumerated in the weight YAML or computed dynamically?
3. **Cache model fidelity**: A full set-associative LRU model adds complexity. An initial P0 can assume all memory traffic hits DRAM (conservative / worst-case Roofline).
4. **DLA descriptor format**: NVDLA layer descriptors are not publicly documented in full. The DLA parser may need to target a simplified subset or the open-source NVDLA spec.

---

*Document version: 0.2 — Consistency and schema update*
*Author: auto-generated via Claude Code*
