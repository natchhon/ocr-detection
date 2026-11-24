"""Reading with BOTH heads (step 2a).

1. The decoder writes its k best guesses (beam search, batched).
2. The CTC draft (fast, column by column) is added as one more guess.
3. Every guess gets two scores: how likely the DECODER finds it, and how likely the CTC head finds it
   (exact CTC probability of the whole text, from F.ctc_loss). Final score = λ·CTC + (1−λ)·decoder.

Why it helps: when the decoder loses its place on a long line it repeats words ("…ធម៌ នមនុស្សធម៌ នមនុស្សធម៌").
The CTC head looks at every column of the picture, so a guess with extra or missing words gets a very low CTC
score and loses. When the CTC draft drops a small vowel, the decoder (which knows real Khmer words) wins.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .tokenizer import BLANK, BOS, EOS, PAD


@torch.no_grad()
def decoder_beam(model, memory, mem_pad, limits, k=4):
    """Beam search with the decoder. limits: max clusters per line (runaway guard).
    Returns, per line, a list of finished guesses (ids without <s>/</s>)."""
    B, dev = memory.shape[0], memory.device
    mem = memory.repeat_interleave(k, 0)
    pad = mem_pad.repeat_interleave(k, 0)
    ids = torch.full((B * k, 1), BOS, device=dev)
    score = torch.full((B, k), float("-inf"), device=dev)
    score[:, 0] = 0  # all beams start equal: keep only one alive at first
    finished = [[] for _ in range(B)]
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    for step in range(int(max(limits))):
        logp = model.decoder(ids, mem, pad)[:, -1].float().log_softmax(-1)  # (B·k, V)
        logp[:, PAD] = logp[:, BOS] = logp[:, BLANK] = float("-inf")
        cand = (score[:, :, None] + logp.view(B, k, -1)).view(B, -1)          # (B, k·V)
        top, idx = cand.topk(2 * k, -1)
        V = logp.shape[-1]
        new_ids, new_score = [], torch.full((B, k), float("-inf"), device=dev)
        rows = []
        for b in range(B):
            kept = 0
            for s, j in zip(top[b].tolist(), idx[b].tolist()):
                beam, tokn = divmod(j, V)
                if s == float("-inf"):
                    break
                if tokn == EOS:
                    if alive[b]:
                        finished[b].append((ids[b * k + beam, 1:].tolist(), s))
                    continue
                if kept < k:
                    rows.append(b * k + beam)
                    new_ids.append(tokn)
                    new_score[b, kept] = s
                    kept += 1
            while kept < k:  # fill dead slots (their score stays -inf)
                rows.append(b * k)
                new_ids.append(PAD)
                kept += 1
            # stop this line when it has k finished guesses better than any guess still being written,
            # or when it reached its length limit
            best_alive = new_score[b].max().item()
            if len(finished[b]) >= k and max(f[1] for f in finished[b]) >= best_alive or step + 1 >= limits[b]:
                alive[b] = False
        if not alive.any():
            break
        ids = torch.cat([ids[rows], torch.tensor(new_ids, device=dev)[:, None]], 1)
        score = torch.where(alive[:, None], new_score, torch.full_like(new_score, float("-inf")))
    return [[f[0] for f in sorted(fs, key=lambda f: -f[1])[:k]] for fs in finished]


@torch.no_grad()
def score_guesses(model, memory, mem_pad, ctc_lp, lens, guesses):
    """guesses: list (one per line) of lists of id-lists. Returns per line a list of (ids, dec_logp, ctc_logp)."""
    flat = [(b, g) for b, gs in enumerate(guesses) for g in gs]
    out = [[] for _ in guesses]
    for i in range(0, len(flat), 256):
        part = flat[i:i + 256]
        bi = torch.tensor([b for b, _ in part], device=memory.device)
        L = max(1, max(len(g) for _, g in part))
        tgt = torch.full((len(part), L), PAD, device=memory.device)
        tl = torch.tensor([len(g) for _, g in part], device=memory.device)
        for r, (_, g) in enumerate(part):
            if g:
                tgt[r, : len(g)] = torch.tensor(g, device=memory.device)
        # decoder: log P(text + </s>)
        dec_in = torch.cat([torch.full((len(part), 1), BOS, device=memory.device), tgt], 1)
        dec_out = torch.cat([tgt, torch.full((len(part), 1), PAD, device=memory.device)], 1)
        dec_out[torch.arange(len(part)), tl] = EOS
        lp = model.decoder(dec_in, memory[bi], mem_pad[bi]).float().log_softmax(-1)
        tok = lp.gather(-1, dec_out.clamp(min=0)[..., None])[..., 0]
        dec = (tok * (dec_out != PAD)).sum(1)
        # CTC: exact log P(text) summed over all alignments
        ctc = -F.ctc_loss(ctc_lp[bi].transpose(0, 1), tgt, lens[bi], tl, blank=BLANK,
                          reduction="none", zero_infinity=False)
        for r, (b, g) in enumerate(part):
            out[b].append((g, dec[r].item(), ctc[r].item()))
    return out


@torch.no_grad()
def joint_read(model, images, widths, k=4, lams=(0.5,)):
    """Read lines with both heads. Returns {λ: [ids per line]} (several λ at once costs nothing extra),
    plus the CTC drafts."""
    memory, mem_pad, ctc_lp, lens = model.encode(images, widths)
    drafts = model.ctc_greedy(ctc_lp, lens)
    limits = [int(1.5 * len(d)) + 4 for d in drafts]
    beams = decoder_beam(model, memory, mem_pad, limits, k)
    guesses = []
    for bs, d in zip(beams, drafts):
        g = [list(x) for x in bs]
        if d not in g:
            g.append(d)
        guesses.append(g)
    scored = score_guesses(model, memory, mem_pad, ctc_lp, lens, guesses)
    res = {}
    for lam in lams:
        res[lam] = [max(s, key=lambda t: lam * t[2] + (1 - lam) * t[1])[0] for s in scored]
    return res, drafts
