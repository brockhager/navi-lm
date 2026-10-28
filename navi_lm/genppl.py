"""Generative perplexity under the same protocol as NCM's LM1B evaluation (`genppl_flm`):
decode BERT ids to text keeping special tokens, rescore the first 128 GPT-2 tokens with a
reference LM, report corpus-level perplexity, and mean per-sample unigram entropy of the
generated BERT ids. Ported from HazyResearch/ncm scripts/eval_merge.py (MIT)."""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _entropy(ids) -> float:
    t = torch.tensor(ids)
    counts = torch.unique(t, return_counts=True)[1].float()
    return float(torch.special.entr(counts / t.numel()).sum())


def genppl_flm(ids_by_seed: dict, bert_name: str = "bert-base-uncased", ref_model: str = "gpt2-large", device: str = "cuda"):
    """ids_by_seed: {seed: list of 128-token id lists}. Returns [(seed, genppl, entropy), ...]."""
    gen_tok = AutoTokenizer.from_pretrained(bert_name)
    rtok = AutoTokenizer.from_pretrained(ref_model)
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token
    rlm = AutoModelForCausalLM.from_pretrained(ref_model).to(device).eval()

    def corpus_ppl(texts):
        enc = rtok(texts, return_tensors="pt", return_token_type_ids=False, truncation=True, padding=True, max_length=128)
        ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        nll_sum = tok_sum = 0.0
        with torch.no_grad():
            for i in range(0, ids.shape[0], 16):
                s_, m_ = ids[i : i + 16], attn[i : i + 16]
                logits = rlm(s_, attention_mask=m_)[0].transpose(-1, -2)
                nll = F.cross_entropy(logits[..., :-1], s_[..., 1:], reduction="none")
                first_eos = (s_ == rtok.eos_token_id).cumsum(-1) == 1
                valid = (first_eos[..., 1:] + (s_ != rtok.eos_token_id)[..., 1:]).to(nll.dtype)
                nll_sum += float((nll * valid).sum())
                tok_sum += float(valid.sum())
        return float(np.exp(nll_sum / tok_sum))

    out = []
    for seed, ids in ids_by_seed.items():
        texts = gen_tok.batch_decode(ids)
        out.append((seed, corpus_ppl(texts), float(np.mean([_entropy(x) for x in ids]))))
    return out


def format_table(rows) -> str:
    lines = [f"{'seed':>5} {'genPPL':>8} {'entropy':>8}"]
    lines += [f"{s:5d} {p:8.2f} {e:8.4f}" for s, p, e in rows]
    g = np.array([r[1] for r in rows])
    e = np.array([r[2] for r in rows])
    sd = g.std(ddof=1) if len(g) > 1 else 0.0
    lines.append(f"\n{'mean':>5} {g.mean():8.2f} {e.mean():8.4f}   (+/- {sd:.2f} over {len(g)} seeds)")
    return "\n".join(lines)
