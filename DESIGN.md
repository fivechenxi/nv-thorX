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

### 2.1 Key Specifications

| Subsystem | Specification |
|-----------|--------------|
| CPU | 12× Arm Cortex-A78AE @ ~3.2 GHz (Hercules-AE, ARMv8.2-A) |
| GPU | NVIDIA Ampere — 2 048 CUDA cores |
| DLA | 2× Deep Learning Accelerator v3.0 |
| Memory | LPDDR5 — 256-bit bus |
| Peak CPU FP64 | ~768 GFLOP/s |
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
| `cpu` | ARMv8.2-A (AArch64) | NEON/SVE SIMD included |
| `gpu` | NVIDIA PTX ISA (virtual) | Mapped to Ampere micro-ops |
| `dla` | NVDLA layer descriptor | Convolution / activation primitives |

---

## 3. Functional Requirements

| ID | Requirement |
|----|------------|
| FR-01 | Accept instruction traces in JSON, assembly text, or PTX format. |
| FR-02 | Decode every instruction into: mnemonic, operands, FLOP count, memory access size. |
| FR-03 | Accumulate per-kernel metrics: total FLOPs, total bytes read, total bytes written. |
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
│  │  (JSON/ASM/  │    │  (per-domain)│    │  (decoded IR)        │  │
│  │   PTX)       │    └──────────────┘    └──────────┬───────────┘  │
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
- Resolves memory operands through the **Cache Simulator** to determine whether data hits cache or goes to DRAM, adding the corresponding bytes to `kernel.bytes_read` or `kernel.bytes_written`.

The cache simulator uses a simplified LRU set-associative model matching Thor X cache parameters.

#### 4.1.3 Performance Counter Store

```
KernelMetrics {
    name:           str
    domain:         Domain          # CPU | GPU | DLA
    instruction_count: int
    total_flops:    float           # summed FLOP weights
    bytes_read:     int             # DRAM reads (post cache)
    bytes_written:  int             # DRAM writes (post cache)
    arithmetic_intensity: float     # computed = total_flops / (bytes_read + bytes_written)
}
```

#### 4.1.4 Roofline Analyzer

For each `KernelMetrics`:

1. Select hardware domain ceilings (see §5.1).
2. Compute **attainable performance**:
   ```
   P_attainable = min(P_peak_compute,  AI × BW_peak)
   ```
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
| Peak FP64 scalar | 768 GFLOP/s | 2 FMA × 12 cores × 3.2 GHz |
| Peak FP32 NEON (128-bit) | 3 072 GFLOP/s | 4× SIMD width |
| Peak FP16 NEON | 6 144 GFLOP/s | 8× SIMD width |
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
ridge_CPU_FP32  = P_peak_CPU_FP32  / BW_CPU_DRAM  ≈ 3 072 / 68   ≈ 45.2 FLOP/byte
ridge_GPU_FP32  = P_peak_GPU_FP32  / BW_GPU_DRAM  ≈ 16 000 / 204 ≈ 78.4 FLOP/byte
ridge_GPU_FP16  = P_peak_GPU_FP16  / BW_GPU_DRAM  ≈ 32 000 / 204 ≈ 156.9 FLOP/byte
```

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

| Mnemonic | FLOPs | Mem bytes |
|----------|-------|-----------|
| `fadd`, `fsub`, `fmul` | 1 | 0 |
| `fmadd`, `fmla` | 2 | 0 |
| `fdiv` | 1 | 0 |
| `ldr` (64-bit) | 0 | 8 |
| `str` (64-bit) | 0 | 8 |
| `ld1` (128-bit NEON) | 0 | 16 |
| `fmla.4s` (NEON 4×FP32) | 8 (4×FMA) | 0 |
| `fcvt` | 1 | 0 |

### 6.2 PTX Weights (examples)

| PTX instruction | FLOPs | Mem bytes |
|-----------------|-------|-----------|
| `add.f32`, `mul.f32` | 1 | 0 |
| `fma.f32` | 2 | 0 |
| `ld.global.f32` | 0 | 4 |
| `ld.shared.f32` | 0 | 4 (shared, not DRAM) |
| `st.global.f32` | 0 | 4 |
| `dp4a` (INT8 dot product) | 8 | 0 |
| `mma.m16n8k16` (FP16 Tensor) | 512 | 0 |

### 6.3 Extension Points

The weight tables are stored in `config/weights_cpu.yaml` and `config/weights_gpu.yaml`, enabling users to override or extend weights without code changes.

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

---

## 8. Output Report Format

### 8.1 JSON Report

```json
{
  "hardware": "Jetson ThorX / Ampere GPU",
  "domain": "gpu",
  "roofline_ceilings": {
    "peak_compute_gflops": 16000,
    "peak_bandwidth_gbps": 204,
    "ridge_point_flop_per_byte": 78.4
  },
  "kernels": [
    {
      "name": "matmul_fp32",
      "instruction_count": 1024,
      "total_flops": 2097152,
      "bytes_dram": 131072,
      "arithmetic_intensity": 16.0,
      "attainable_gflops": 3264.0,
      "bound": "memory",
      "utilization_pct": 20.4
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

*Document version: 0.1 — Initial draft*
*Author: auto-generated via Claude Code*
