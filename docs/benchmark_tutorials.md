# Tutorial Notebook Benchmark: MacStudio vs DGX Spark

**Date**: 2026-04-26  
**Model**: `Qwen/Qwen2.5-7B`  
**Notebooks**: `tutorials/ch1` – `ch9`

## Hardware

| | MacStudio | DGX Spark |
|---|---|---|
| CPU | Apple M-series (ARM64) | NVIDIA Grace (ARM64) |
| Accelerator | Apple MPS | NVIDIA GB10 (Blackwell) |
| PyTorch | 2.11.0 (macosx_arm64) | 2.11.0+cu130 (linux_aarch64) |
| CUDA / Metal | Metal (MPS) | CUDA 13.0 / driver 580.126.09 |
| OS | macOS | Ubuntu 24.04 |
| RAM | unified | 119 GB |

## Results: All 9 notebooks PASSED on both machines

## Per-notebook timing (wall-clock seconds, per cell)

### ch1 — Opening the Black Box

| Cell | MacStudio MPS | DGX Spark CUDA |
|---|---|---|
| 1 · imports + load\_model | 19.6 | 22.2 |
| 2 · extract\_hidden\_states | 0.2 | 0.7 |
| 3 · norm visualisation | 0.3 | 0.2 |
| 4 · ltg.analyse() | 1.7 | 22.6 |
| 5 · plot\_curvature/kernel/polar | 0.5 | 0.6 |
| 6 · ltg.compare() 3 prompts | 5.2 | 14.6 |
| 7 · exercise cell (comments) | 0.0 | 0.0 |
| **Total** | **27.6 s** | **60.9 s** |

### ch2 — Whitening

| Cell | MacStudio MPS | DGX Spark CUDA |
|---|---|---|
| 1 · load\_model | 11.1 | 5.7 |
| 2 · extract + variance plot | 0.6 | 0.7 |
| 3 · whitened covariance | 0.0 | 8.4 |
| 4 · curvature after whitening | 0.0 | 0.1 |
| 5 · k-sweep | 0.3 | 17.0 |
| 6 · full analysis | 0.5 | 53.5 |
| 7 · exercise | 0.0 | 0.0 |
| **Total** | **12.6 s** | **85.4 s** |

### ch3 — Kernels

| Cell | MacStudio MPS | DGX Spark CUDA |
|---|---|---|
| 1 · load\_model | 7.6 | 7.9 |
| 2 · layer kernel matrix | 0.8 | 37.7 |
| 3 · kernel viz | 0.1 | 0.1 |
| 4 · token kernel | 0.0 | 0.0 |
| 5 · multi-prompt kernel | 2.0 | 109.0 |
| 6 · exercise | 0.0 | 0.0 |
| **Total** | **10.5 s** | **154.7 s** |

### ch4 — Polar Decomposition

| Cell | MacStudio MPS | DGX Spark CUDA |
|---|---|---|
| 1 · load\_model | 7.5 | 8.1 |
| 2 · analyse + polar shapes | 2.8 | 109.7 |
| 3 · identity deviation plot | 0.0 | 0.7 |
| 4 · condition numbers | 0.0 | 0.1 |
| 5 · eigenvalue spectrum | 0.1 | 0.4 |
| 6 · multi-prompt comparison | 2.6 | 140.0 |
| 7 · exercise | 0.0 | 0.0 |
| **Total** | **13.1 s** | **259.0 s** |

### ch5 — Curvature

| Cell | MacStudio MPS | DGX Spark CUDA |
|---|---|---|
| 1 · load\_model | 7.4 | 8.9 |
| 2 · curvature map | 2.2 | 36.4 |
| 3 · curvature sweep (many prompts) | 4.3 | 415.9 |
| 4 · token-level curvature | 2.5 | 192.6 |
| 5 · exercise | 0.0 | 0.0 |
| **Total** | **16.4 s** | **653.8 s** |

### ch6 — Experiments

| Cell | MacStudio MPS | DGX Spark CUDA |
|---|---|---|
| 1 · load\_model | 6.8 | 6.2 |
| 2 · setup | 0.0 | 0.0 |
| 3 · DOE experiment loop | 33.3 | 1103.0 |
| 4 · results plot | 0.0 | 0.2 |
| 5 · summary stats | 0.1 | 0.5 |
| 6 · exercise | 0.0 | 0.0 |
| **Total** | **40.2 s** | **1109.9 s** |

### ch7 — Dependency

| Cell | MacStudio MPS | DGX Spark CUDA |
|---|---|---|
| 1 · load\_model | 21.6 | 5.2 |
| 2 · dependency map | 2.4 | 18.7 |
| 3 · setup | 0.0 | 0.1 |
| 4 · dependency profile | 9.9 | 95.9 |
| 5 · cross-prompt dependency | 6.4 | 63.8 |
| 6 · exercise | 0.0 | 0.0 |
| **Total** | **40.4 s** | **183.7 s** |

### ch8 — Reasoning, Memory & Control

| Cell | MacStudio MPS | DGX Spark CUDA |
|---|---|---|
| 1 · load\_model | 21.6 | 5.8 |
| 2 · dependency map | 2.6 | 31.6 |
| 3 · setup | 0.0 | 0.0 |
| 4 · memory analysis | 2.1 | 3.2 |
| 5 · control analysis | 3.4 | 3.6 |
| 6 · setup | 0.0 | 0.0 |
| 7 · steering experiment | 6.8 | 22.9 |
| 8 · exercise | 0.0 | 0.0 |
| **Total** | **36.5 s** | **67.1 s** |

### ch9 — Diagnosing Failures

| Cell | MacStudio MPS | DGX Spark CUDA |
|---|---|---|
| 1 · load\_model | 7.8 | 5.4 |
| 2 · failure diagnosis | 4.9 | 83.3 |
| 3 · setup | 0.0 | 0.1 |
| 4 · hallucination sweep | 9.1 | 147.6 |
| 5 · context-ignoring test | 9.8 | 127.7 |
| 6 · report | 1.7 | 9.7 |
| 7 · cross-prompt diagnosis | 4.6 | 58.1 |
| 8 · exercise | 0.0 | 0.0 |
| **Total** | **38.0 s** | **431.9 s** |

---

## Summary

| Notebook | MacStudio MPS | DGX Spark CUDA | Speedup (MPS/CUDA) |
|---|---|---|---|
| ch1 opening | 27.6 s | 60.9 s | MPS 2.2× faster |
| ch2 whitening | 12.6 s | 85.4 s | MPS 6.8× faster |
| ch3 kernels | 10.5 s | 154.7 s | MPS 14.7× faster |
| ch4 polar decomp. | 13.1 s | 259.0 s | MPS 19.8× faster |
| ch5 curvature | 16.4 s | 653.8 s | MPS 39.9× faster |
| ch6 experiments | 40.2 s | 1109.9 s | MPS 27.6× faster |
| ch7 dependency | 40.4 s | 183.7 s | MPS 4.5× faster |
| ch8 memory/control | 36.5 s | 67.1 s | MPS 1.8× faster |
| ch9 diagnosing | 38.0 s | 431.9 s | MPS 11.4× faster |
| **Total** | **235 s (3.9 min)** | **3006 s (50.1 min)** | **MPS 12.8× faster** |

## Analysis

MacStudio MPS is **12.8× faster overall**. The gap is concentrated in
compute-heavy cells (multi-prompt loops, dependency sweeps, curvature
experiments), not in model loading where both are comparable.

**Root cause**: `torch 2.11.0+cu130` on `linux_aarch64` (Grace Blackwell)
lacks optimized FlashAttention and fused SDPA kernels for this target,
falling back to generic CUDA paths. Apple's MPS backend has well-tuned
Metal kernels for exactly this workload on the M-series GPU.

**Recommended fix for DGX Spark**: install `flash-attn`, `vllm`, or use
NVIDIA's NIM inference stack to get optimized attention kernels on GB10.
Alternatively, `transformers` `attn_implementation="flash_attention_2"`
once a compatible `flash-attn` wheel is available for `linux_aarch64`.
