"""Train the matched autoregressive baseline on tokenized LM1B (NCM release format).

Example (quick check, about 2M tokens):
  python scripts/train_ar.py --out_dir runs/ar_smoke --max_tokens 2000000 --eval_every 20
Example (real run, 200M tokens):
  python scripts/train_ar.py --out_dir runs/ar_s42 --seed 42 --max_tokens 200000000 --compile
"""
import argparse
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from navi_lm.ar_model import ARConfig, ARLM
from navi_lm.data import SEQ_LEN, Lm1bChunks, make_batch


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="~/ncm/data/lm1b")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=200_000_000, help="training token budget")
    p.add_argument("--global_bs", type=int, default=128, help="sequences per optimizer step")
    p.add_argument("--micro_bs", type=int, default=16, help="sequences per forward pass")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_chunks", type=int, default=4096, help="valid chunks used for periodic evals")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def lr_at(step, total, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    prog = (step - a.warmup) / max(1, total - a.warmup)
    return a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, data, n_chunks, micro_bs, dev):
    model.eval()
    n = len(data) if n_chunks <= 0 else min(n_chunks, len(data))
    total, count = 0.0, 0
    for i in range(0, n, micro_bs):
        x = data.get(np.arange(i, min(i + micro_bs, n))).to(dev)
        inp, tgt = make_batch(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(inp, tgt)
        total += float(loss) * tgt.numel()
        count += tgt.numel()
    model.train()
    return total / count


def save(path, model, a, step, tokens):
    torch.save({"model": model.state_dict(), "cfg": asdict(model.cfg), "args": vars(a), "step": step, "tokens": tokens}, path)


def main():
    a = get_args()
    assert a.global_bs % a.micro_bs == 0
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    torch.set_float32_matmul_precision("high")

    train, valid = Lm1bChunks(a.data_dir, "train"), Lm1bChunks(a.data_dir, "valid")
    tokens_per_step = a.global_bs * SEQ_LEN
    total_steps = a.max_tokens // tokens_per_step
    need = total_steps * a.global_bs
    assert need <= len(train), f"budget needs {need} chunks, train has {len(train)}"
    order = rng.permutation(len(train))[:need]

    model = ARLM(ARConfig()).to(dev)
    n_params = model.num_params()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, 0.95), fused=True,
    )
    fwd = torch.compile(model) if a.compile else model
    print(f"params {n_params / 1e6:.1f}M | steps {total_steps} | tokens/step {tokens_per_step} | budget {total_steps * tokens_per_step / 1e6:.0f}M tokens")

    log = open(os.path.join(a.out_dir, "metrics.jsonl"), "a")
    accum = a.global_bs // a.micro_bs
    train_secs, steady_secs, steady_tokens, run_loss, run_n = 0.0, 0.0, 0, 0.0, 0
    WARM = 20  # steps excluded from the steady-state throughput (compile / cudnn warmup)
    model.train()
    for step in range(total_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, a)
        step_loss = torch.zeros((), device=dev)
        for m in range(accum):
            lo = step * a.global_bs + m * a.micro_bs
            x = train.get(order[lo : lo + a.micro_bs]).to(dev, non_blocking=True)
            inp, tgt = make_batch(x)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = fwd(inp, tgt)
            (loss / accum).backward()
            step_loss += loss.detach() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        train_secs += dt
        if step >= WARM:
            steady_secs += dt
            steady_tokens += tokens_per_step
        run_loss += float(step_loss)
        run_n += 1
        s1 = step + 1
        if s1 % a.log_every == 0 or s1 == total_steps:
            rec = {"step": s1, "tokens": s1 * tokens_per_step, "train_loss": run_loss / run_n, "lr": opt.param_groups[0]["lr"],
                   "tok_per_sec": (steady_tokens / steady_secs) if steady_secs else None,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
            run_loss, run_n = 0.0, 0
            print(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
        if s1 % a.eval_every == 0 and s1 != total_steps:
            vl = evaluate(model, valid, a.eval_chunks, a.micro_bs, dev)
            rec = {"step": s1, "tokens": s1 * tokens_per_step, "val_loss": vl, "val_ppl": math.exp(vl)}
            print(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
        if s1 % a.save_every == 0:
            save(os.path.join(a.out_dir, "ckpt.pt"), model, a, s1, s1 * tokens_per_step)

    vl = evaluate(model, valid, 0, a.micro_bs, dev)  # full validation set
    summary = {"params": n_params, "steps": total_steps, "tokens": total_steps * tokens_per_step, "seed": a.seed,
               "val_loss_full": vl, "val_ppl_full": math.exp(vl), "train_secs": train_secs,
               "tok_per_sec_steady": (steady_tokens / steady_secs) if steady_secs else None,
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "compile": a.compile}
    print(json.dumps(summary)); log.write(json.dumps({"final": summary}) + "\n"); log.close()
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2)
    save(os.path.join(a.out_dir, "ckpt.pt"), model, a, total_steps, total_steps * tokens_per_step)


if __name__ == "__main__":
    main()
