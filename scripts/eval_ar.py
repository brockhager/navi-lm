"""Sample from a trained AR checkpoint and score it with NCM's LM1B protocol
(5 seeds x 256 samples, generative perplexity under GPT-2 large, unigram entropy).

  python scripts/eval_ar.py --ckpt runs/ar_s42/ckpt.pt
"""
import argparse
import gc
import json
import os

import torch

from navi_lm.ar_model import ARConfig, ARLM
from navi_lm.genppl import format_table, genppl_flm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    a = p.parse_args()

    dev = "cuda"
    ck = torch.load(a.ckpt, map_location="cpu")
    model = ARLM(ARConfig(**ck["cfg"])).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    ids_by_seed = {}
    for seed in a.seeds:
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        chunks = []
        for i in range(0, a.n_samples, a.batch):
            chunks.append(model.generate(min(a.batch, a.n_samples - i), temperature=a.temperature, generator=g).cpu())
        ids_by_seed[seed] = torch.cat(chunks).tolist()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    rows = genppl_flm(ids_by_seed)
    print(format_table(rows))
    out = os.path.join(os.path.dirname(a.ckpt), "gen_eval.json")
    json.dump({"rows": rows, "tokens": ck["tokens"], "step": ck["step"], "temperature": a.temperature}, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
