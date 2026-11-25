"""RL pair on the shared RL kernel set: `truestar rlpair examples/rl_kernels_pair.py --device cuda
--kernels rl --strict --tp 2` exits 0 (bitwise equal logprobs, batch invariant sampler, TP 1 equal
to a simulated TP 2).

The trainer is a teacher forced forward over the whole sequences of a small Llama style model
(RMSNorm, rotary, grouped query attention, SwiGLU; random weights, built in code). The sampler
batches the two rows with a longer third row, so they are left padded by 3, prefills the prompts
into a KV cache with a key padding mask and decodes one token per step. Without `--kernels rl`
the same pair runs on PyTorch's own kernels and is not bitwise equal.
"""

import torch

from truestar.rlkernels import llama as L

PROMPT = 6
EXTRA = 3  # the third row is this much longer, so the trainer's rows are left padded by it


def get_model():
    return L.make_model(seed=0, device="cuda" if torch.cuda.is_available() else "cpu")


def get_tokens():
    g = torch.Generator().manual_seed(7)
    return torch.randint(0, L.Config().vocab, (2, 18), generator=g)


def get_prompt_len():
    return PROMPT


def get_layout():
    return {"rows": [0, 1], "left_pad": [EXTRA, EXTRA]}


def sample_logits(model, tokens, prompt_len):
    g = torch.Generator().manual_seed(11)
    extra = torch.randint(0, model.c.vocab, (tokens.shape[1] + EXTRA,), generator=g)
    rows = [r.cpu() for r in tokens] + [extra]
    lps = L.decode_logprobs(model, rows, [prompt_len] * len(tokens) + [prompt_len + EXTRA])
    T = tokens.shape[1]
    return torch.stack([lp[prompt_len - 1:T - 1] for lp in lps[:len(tokens)]])


def tp_logits(model, tokens, tp):
    """Logits of the trainer forward with weights split over `tp` simulated ranks (one device)."""
    return L.tp_train_logits(model, tokens, tp)
