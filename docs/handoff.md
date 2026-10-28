# Project NaviLM (`navi-lm`)

Research project: build and experiment with a more compute-efficient language model architecture.
Handoff date: 2026-10-28. Chosen name: NaviLM (repo `navi-lm`, Python import `navi_lm`). On 2026-10-28 I searched GitHub: no repo named `navi-lm` and no user or org with that exact name. Plain `navi` is heavily used (including a dormant small-LM repo, `SkuldNorniern/navi`). PyPI and domains are not checked.

## Hardware

| Item | Value |
|---|---|
| GPU | RTX 5060, 8 GB VRAM (~7 GB free), Blackwell, driver 610.88, CUDA 13.3 |
| RAM | ~16 GB |
| Disk | 88 GB free on C: |
| Plan | WSL2 Ubuntu, PyTorch with CUDA 12.8+ (needed for Blackwell), WSL memory capped at ~10-12 GB |
| Ceiling | ~50-150M param models; LM1B / TinyStories / FineWeb-Edu slices. Rent a GPU only for larger runs. |

## Background

**Next-Concept Modeling (NCM)**, Stanford Hazy Research + Together AI.
- Paper: https://arxiv.org/abs/2610.14162
- Code (MIT): https://github.com/HazyResearch/ncm
- A multiscale residual VQ-VAE encodes text into a hierarchy of discrete codes. A transformer generates one level at a time, all codes in a level in parallel.
- Claims 10-31x faster generation than strong diffusion LMs and ~4-5x fewer training tokens (LM1B, OpenWebText).
- Scaling results are preliminary (LM1B, 97M to 725M generators).
- Not the same as Meta's Large Concept Models (sentence-level, SONAR space).

**Efficiency families**: MoE, linear attention / SSMs (Mamba 2, Gated DeltaNet), sparse/GQA attention, hybrids, diffusion LMs, MossNet.
Survey: https://arxiv.org/abs/2508.09834

| Family | What it saves |
|---|---|
| MoE | Training compute (FLOPs per token); all experts still sit in memory |
| SSM / linear attention / hybrids | Long-context inference |
| Diffusion LMs, NCM | Generation latency (fewer sequential steps) |

## Corrections to the earlier handoff (checked against the paper and repo)

- The handoff said NCM was not compared against autoregressive LMs. The paper does include AR baselines: TinyStories, OpenWebText, XSum, WMT14 and Sudoku. There is no AR baseline on LM1B, and no comparison with speculative decoding.
- The public repo looks evaluation-focused: `evaluate.py`, generation/scoring scripts, `models/`, `tokenization/`, and released checkpoints on Hugging Face. Confirmed on 2026-10-28 by listing `models/`, `tokenization/`, `scripts/` and `utils/`: model and tokenizer definitions plus eval scripts only, no training loop, optimizer or loss code. I will need to write the training loop myself (the generator loss is in the paper, section 3.4; the tokenizer is released frozen). "Reproduce the smallest LM1B result" therefore means evaluating the released LM1B small checkpoint, unless training code is added or I write it.
- LM1B training cost in the paper: tokenizer (~111M params) plus generator (~97M params), ~13.4B token positions total. That is likely too heavy for the 8 GB card at paper settings. Plan on TinyStories-scale runs locally.

## Plan

1. Set up WSL2 + PyTorch, confirm the GPU is visible. See `SETUP.md`.
2. Clone `HazyResearch/ncm`, evaluate the released smallest LM1B checkpoint. Confirm whether training code exists.
3. Pick ONE variable and change only that. See `EXPERIMENTS.md`.
4. Report loss and tokens/sec at equal compute, 3 seeds.

## Open decisions

- First experiment: leaning matched autoregressive baseline.
- PyPI and domain availability for `navi-lm`. I could not query PyPI directly, so confirm with `pip index versions navi-lm`.
