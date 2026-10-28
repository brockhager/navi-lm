"""LM1B as fixed 128-token chunks (BERT uncased ids), using the NCM release format.

Files: {split}_bert_30522.bin (uint16 token stream) and {split}_bert_30522_offsets.npy
(chunk start offsets, every 128 tokens). Chunks are consecutive sentences joined by [CLS].
"""
import os

import numpy as np
import torch

SEQ_LEN = 128
BOS_ID = 0  # [PAD] in the BERT vocab; NCM also uses id 0 as its prefix token


class Lm1bChunks:
    def __init__(self, data_dir: str, split: str):
        path = os.path.join(os.path.expanduser(data_dir), f"{split}_bert_30522.bin")
        flat = np.memmap(path, dtype=np.uint16, mode="r")
        assert flat.size % SEQ_LEN == 0, f"{path}: size {flat.size} is not a multiple of {SEQ_LEN}"
        self.x = flat.reshape(-1, SEQ_LEN)

    def __len__(self) -> int:
        return self.x.shape[0]

    def get(self, idx) -> torch.Tensor:
        """Rows `idx` as an int64 tensor (B, 128). Sorted reads are faster on a memmap."""
        idx = np.sort(np.asarray(idx))
        return torch.from_numpy(self.x[idx].astype(np.int64))


def make_batch(tokens: torch.Tensor):
    """Autoregressive inputs/targets: predict all 128 tokens from a BOS-prefixed shift."""
    bos = torch.full((tokens.size(0), 1), BOS_ID, dtype=tokens.dtype, device=tokens.device)
    inputs = torch.cat([bos, tokens[:, :-1]], dim=1)
    return inputs, tokens
