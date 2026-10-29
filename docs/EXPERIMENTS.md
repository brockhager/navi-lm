# Experiments

Rule: change ONE variable per experiment. Report loss and tokens/sec at equal compute, 3 seeds.

## Candidates

| # | Experiment | Variable | Status |
|---|---|---|---|
| 1 | Matched autoregressive baseline | Generation paradigm (AR vs NCM) | First AR run done (200M tokens, 1 seed). Not yet equal compute with NCM; see run log. |
| 2 | SSM backbone (Mamba 2 / Gated DeltaNet) for the level predictor | Backbone | After #1 |
| 3 | Quantizer / level count in the tokenizer | Tokenizer | Candidate |
| 4 | MoE backbone for the level predictor | Backbone | Low priority at this scale |
| 5 | Coarse levels in parallel, then autoregressive refinement | Generation schedule | Candidate |

## Why #1 first

- The AR baseline is the comparison the NCM paper does not give on LM1B.
- Every later variable needs the baseline to mean anything.
- Cheap on 8 GB VRAM at ~50-150M params.

## Notes

- Paper ablations worth reusing (TinyStories, 12 levels): level count K in {4, 8, 16, 32}, level schedule (linear, quadratic, sqrt, log, exponential, cosine), block-diagonal vs block-causal attention, conditioning corruption rate.
- MoE saves FLOPs per token but not memory, so it buys little at 50-150M params.
- Sparse attention / GQA gives negligible gain at short LM1B sequences (L=128).

## Run log

| Date | Experiment | Seed | Loss | Tokens/sec | Notes |
|---|---|---|---|---|---|
| 2026-10-28 | Reproduce NCM LM1B small (released checkpoint, eval only) | 42,0,1,2,3 | n/a | not measured | Gen. PPL 93.38 (+/- 1.62), entropy 4.313, 1280 samples, RTX 5060. Paper: 93.05 / 4.31. Matches. |
| 2026-10-29 | AR baseline, 108.5M params, 200M tokens (train) | 42 | val 3.723 (ppl 41.4) | 37,943 | 88 min, 4.2 GB peak, bf16 + compile. |
| 2026-10-29 | AR baseline, generation score (same protocol as NCM row) | 42,0,1,2,3 | n/a | n/a | Gen. PPL 122.20 (+/- 2.24), entropy 4.326, 1280 samples, temperature 1.0. Training seed is 42 only. NCM generator saw about 6.5B token positions (13.4B with tokenizer), so the AR run got roughly 30x fewer. Not an equal-compute comparison. |
