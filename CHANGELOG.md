# Changelog

A running log of what was done, newest first. Results live in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## 2026-10-28

### Repos
- Published the public repo `navi-lm` (MIT) with the README, license, `.gitignore` and the docs in `docs/`.
- Created a private companion repo for scratch work and raw logs.
- Chose the name NaviLM. Checked GitHub for `navi-lm` (repo, user and org names): no exact match. PyPI and domains are not yet checked.

### Environment
- Installed and updated WSL2 (WSL 3.0.3) with Ubuntu 24.04 (Python 3.12). Capped WSL memory at 11 GB with 8 GB swap.
- Confirmed the RTX 5060 (8 GB, Blackwell) is visible inside WSL via `nvidia-smi`.
- Created a Python venv and installed PyTorch `2.11.0+cu128`. `torch.cuda.is_available()` is `True`, device capability `(12, 0)`.
- Cloned [HazyResearch/ncm](https://github.com/HazyResearch/ncm) and installed its requirements. The CUDA build of torch was left intact.

### Reproduction
- Downloaded the released LM1B checkpoints (about 5 GB: shared tokenizer plus small, medium and large generators).
- Evaluated the small checkpoint with the repo's `evaluate.py`: mean Gen. PPL **93.38** (+/- 1.62 over 5 seeds, 1,280 samples), entropy **4.313**. The paper reports 93.05 and 4.31, so the reproduction matches.

### Findings
- The NCM repo contains model and tokenizer definitions plus evaluation scripts, with no training loop, optimizer or loss code. The training loop has to be written from the paper (generator loss in section 3.4; the tokenizer is released frozen).
- The paper does include autoregressive baselines on TinyStories, OpenWebText, XSum, WMT14 and Sudoku, but not on LM1B. A matched AR baseline on LM1B is the first planned experiment.

### Docs
- Added the project docs: index, summary, handoff and background, setup checklist, and experiment plan with a run log.

### Data
- Inspected the released tokenized LM1B data: `uint16` BERT-uncased token streams in fixed 128-token chunks (7,087,759 train, 71,825 valid; about 907M training tokens), with sentences joined by `[CLS]`.

### Code (written, not yet run)
- `navi_lm/ar_model.py`: decoder-only AR baseline sized like the NCM LM1B generator (12 layers, 12 heads, width 768, RoPE, dropout 0.1).
- `navi_lm/data.py`: memory-mapped chunk loader and the BOS-shifted inputs/targets.
- `navi_lm/genppl.py`: generative perplexity and entropy, ported from NCM's `genppl_flm` protocol so AR and NCM numbers are comparable.
- `scripts/train_ar.py`: training loop with a token budget, bf16, cosine schedule, validation loss, tokens/sec and peak memory logging.
- `scripts/eval_ar.py`: samples 5 seeds x 256 and scores them with GPT-2 large.

### Next
- Smoke-test the training script on a 2M-token budget.
- Measure tokens/sec and VRAM, then pick the token budget for the real AR run.
- Run the AR baseline and compare with NCM at equal compute.
