# TurboQuant: KV Cache Quantization for SGLang

## Overview

This project integrates **TurboQuant** — an online vector quantization method for KV cache compression — into SGLang, targeting Qwen3.5 models.

TurboQuant achieves near-optimal distortion rates for both MSE and inner product metrics, enabling KV cache compression to **2.5-3.5 bits per channel** with negligible quality loss. At 3.5 bits, it matches full-precision quality exactly on LongBench benchmarks.

**TurboQuant is a KV cache quantization method, NOT a weight quantization method.** It is complementary to weight quantization (GPTQ, AWQ, FP8, etc.) — both can be used together.

## Target Models

- **Qwen3.5-9B** (dense, hybrid DeltaNet + full attention)
- **Qwen3.5-27B** (dense, hybrid DeltaNet + full attention)
- **Qwen3.5-35B-A3B** (MoE, 256 experts, hybrid DeltaNet + full attention)

TurboQuant only applies to **full attention layers** (every 4th layer in Qwen3.5). DeltaNet/linear attention layers use internal recurrent state and have no KV cache.

## Key Results (from paper, Llama-3.1-8B-Instruct)

| Method | KV Bits | LongBench Avg |
|--------|---------|---------------|
| Full Cache | 16 | 50.06 |
| KIVI | 3 | 48.50 |
| KIVI | 5 | 50.16 |
| PolarQuant | 3.9 | 49.78 |
| **TurboQuant** | **2.5** | **49.44** |
| **TurboQuant** | **3.5** | **50.06** |

Needle-in-a-haystack: 0.997 (identical to full precision) at 4x compression.

## References

- **Paper:** [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- **Authors:** Amir Zandieh (Google Research), Majid Daliri (NYU), Majid Hadian (Google DeepMind), Vahab Mirrokni (Google Research)
- **QJL repo:** [github.com/amirzandieh/QJL](https://github.com/amirzandieh/QJL) — CUDA kernels for the QJL component
- **PolarQuant repo:** [github.com/ericshwu/PolarQuant](https://github.com/ericshwu/PolarQuant) — Triton kernels for polar coordinate quantization
- **MLX-VLM PR:** [Blaizzy/mlx-vlm#858](https://github.com/Blaizzy/mlx-vlm/pull/858) — Reference implementation for Apple MLX (confirmed faithful to paper)

## Documentation

- [Paper Summary](PAPER_SUMMARY.md) — Complete mathematical specification from the paper
- [SGLang Architecture](SGLANG_ARCHITECTURE.md) — Integration points in SGLang's codebase
- [Qwen3.5 Notes](QWEN3_5_NOTES.md) — Model architecture and TurboQuant implications
- [Component References](COMPONENT_REFERENCES.md) — QJL, PolarQuant, and MLX code mapping
- [Implementation Roadmap](IMPLEMENTATION_ROADMAP.md) — Phased implementation plan
