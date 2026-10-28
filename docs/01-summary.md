# Project NaviLM: Summary

**Goal:** build and experiment with a more compute-efficient language model architecture. Name: NaviLM, repo `navi-lm` (no GitHub repo, user or org with that exact name as of 2026-10-28; PyPI and domains unchecked).

**Hardware:** RTX 5060 (8 GB VRAM), ~16 GB RAM, 88 GB free disk. WSL2 Ubuntu + PyTorch CUDA 12.8+. Realistic ceiling ~50-150M params.

**Base work:** Next-Concept Modeling (NCM, Stanford Hazy Research + Together AI). A multiscale residual VQ-VAE turns text into a coarse-to-fine hierarchy of discrete codes. A transformer predicts one level at a time, all codes in parallel. Claims 10-31x faster generation than strong diffusion LMs and ~4-5x fewer training tokens. Scaling results are preliminary.

**Key findings**
- The paper does include AR baselines (TinyStories, OWT, XSum, WMT14, Sudoku), but none on LM1B and no speculative-decoding comparison.
- The public repo looks evaluation-only (released checkpoints, `evaluate.py`). Training code was not visible.
- LM1B training at paper settings (~13.4B token positions) is likely too heavy for the local GPU.

**Plan**
1. WSL2 + PyTorch, confirm GPU.
2. Clone `HazyResearch/ncm`, evaluate the released LM1B small checkpoint.
3. One-variable experiment, leading pick: matched autoregressive baseline. Then an SSM backbone.
4. Report loss and tokens/sec at equal compute, 3 seeds.

**Open:** first experiment confirmation, PyPI/domain check for `navi-lm`, whether training code exists.

See [00-index.md](00-index.md) for all documents.
