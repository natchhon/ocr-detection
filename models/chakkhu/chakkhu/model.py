"""Chakkhu: a small Khmer + English text-line reader (≈ 17M parameters).

    line image (1 × 64 × W, grey)
      → CNN "eyes"          ResNet blocks, height 64 → 2 rows, width ÷ 8  (keeps what is above / below)
      → Transformer encoder 6 layers, d 320, sees the WHOLE line (2 rows × W/8 positions)
      ├→ CTC head           per column: a cluster or <blank>  → fast draft, letter count, runaway guard, confidence
      └→ cluster decoder    3 layers, writes one Khmer CLUSTER per step (ស្ត្រី = 1 step), in typed order;
                            pre-trained as a language model on Khmer text first (memory=None), so it knows real words

Training loss = 0.3 · CTC + 0.7 · cross-entropy (both heads learn together).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import BLANK, BOS, EOS, PAD


@dataclass
class Config:
    vocab: int = 3402
    height: int = 64        # input line height in pixels
    d: int = 320            # model width
    heads: int = 8
    ffn: int = 1280
    enc_layers: int = 6
    dec_layers: int = 3
    max_cols: int = 512     # widest line: 512 × 8 = 4096 px at height 64
    max_len: int = 256      # longest text, in clusters
    dropout: float = 0.1
    ctc_weight: float = 0.3
    smooth: float = 0.1     # label smoothing of the decoder loss


# ---- 1. CNN "eyes" ------------------------------------------------------------------------------
class Block(nn.Module):
    """Residual block: two 3×3 convs; `stride` shrinks (height, width)."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = nn.Sequential() if stride == (1, 1) and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.gelu(self.b2(self.c2(F.gelu(self.b1(self.c1(x))))) + self.skip(x))


class CNN(nn.Module):
    """64 × W  →  2 rows × W/8 columns × d.  Width is cut only 3 times (thin English letters survive),
    height 5 times (64 → 2), so the encoder still knows what sits above and below the line."""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            Block(32, 64, (2, 2)),     # 32 × W/2
            Block(64, 96, (2, 2)),     # 16 × W/4
            Block(96, 96, (1, 1)),
            Block(96, 160, (2, 2)),    # 8 × W/8
            Block(160, 160, (1, 1)),
            Block(160, 256, (2, 1)),   # 4 × W/8
            Block(256, d, (2, 1)),     # 2 × W/8
        )

    def forward(self, x):
        return self.net(x)


# ---- 2. Transformer encoder ---------------------------------------------------------------------
def sinusoid(n, d):
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class Encoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.register_buffer("col_pos", sinusoid(c.max_cols, c.d), persistent=False)  # any line width
        self.row_pos = nn.Parameter(torch.zeros(2, c.d))                               # top / bottom row
        layer = nn.TransformerEncoderLayer(c.d, c.heads, c.ffn, c.dropout, F.gelu, batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, c.enc_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(c.d)

    def forward(self, f, col_mask):
        """f: (B, d, 2, T) from the CNN; col_mask: (B, T) True = padding column. → (B, 2T, d)"""
        B, d, R, T = f.shape
        x = f.permute(0, 2, 3, 1) + self.col_pos[:T] + self.row_pos[:, None, :]   # (B, 2, T, d)
        x = x.reshape(B, R * T, d)
        mask = col_mask.repeat(1, R)                                              # same padding in both rows
        return self.norm(self.layers(x, src_key_padding_mask=mask)), mask


# ---- 3b. Cluster decoder ------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """Pre-norm: self-attention (looks back at clusters already written) → cross-attention (looks at the
    picture; skipped when memory is None = text-only language-model pre-training) → feed-forward."""

    def __init__(self, c: Config):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(c.d), nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.self_attn = nn.MultiheadAttention(c.d, c.heads, c.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(c.d, c.heads, c.dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(c.d, c.ffn), nn.GELU(), nn.Dropout(c.dropout), nn.Linear(c.ffn, c.d))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, y, causal, pad, memory=None, mem_pad=None):
        h = self.n1(y)
        y = y + self.drop(self.self_attn(h, h, h, attn_mask=causal, key_padding_mask=pad, need_weights=False)[0])
        if memory is not None:
            h = self.n2(y)
            y = y + self.drop(self.cross_attn(h, memory, memory, key_padding_mask=mem_pad, need_weights=False)[0])
        return y + self.drop(self.ff(self.n3(y)))


class Decoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.emb = nn.Embedding(c.vocab, c.d, padding_idx=PAD)
        # small start values: the output layer reuses these weights, big ones would give huge first losses
        nn.init.normal_(self.emb.weight, std=c.d ** -0.5)
        with torch.no_grad():
            self.emb.weight[PAD].zero_()
        self.pos = nn.Embedding(c.max_len + 2, c.d)
        self.layers = nn.ModuleList(DecoderLayer(c) for _ in range(c.dec_layers))
        self.norm = nn.LayerNorm(c.d)
        self.scale = math.sqrt(c.d)

    def forward(self, ids, memory=None, mem_pad=None):
        """ids: (B, L) starting with <s>. → logits (B, L, vocab) for the NEXT cluster at each position."""
        L = ids.shape[1]
        y = self.emb(ids) * self.scale + self.pos(torch.arange(L, device=ids.device))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=ids.device), 1)
        for layer in self.layers:
            y = layer(y, causal, ids == PAD, memory, mem_pad)
        return self.norm(y) @ self.emb.weight.T   # output layer shares the embedding weights


# ---- the whole model ----------------------------------------------------------------------------
class Chakkhu(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.cnn = CNN(c.d)
        self.encoder = Encoder(c)
        self.ctc = nn.Linear(c.d, c.vocab)   # 3a: per column, a cluster or <blank>
        self.decoder = Decoder(c)            # 3b

    def encode(self, images, widths):
        """images: (B, 1, 64, W) in [0, 1], white text background padded on the right; widths: real widths in px.
        → memory (B, 2T, d), its padding mask, CTC log-probs (B, T, vocab), CTC lengths (B,)"""
        f = self.cnn(images * 2 - 1)
        T = f.shape[-1]
        lens = torch.clamp((widths + 7) // 8, 1, T)
        col_mask = torch.arange(T, device=images.device)[None, :] >= lens[:, None]
        memory, mem_pad = self.encoder(f, col_mask)
        cols = memory.view(memory.shape[0], 2, T, -1).mean(1)       # merge the 2 rows → one frame per column
        return memory, mem_pad, self.ctc(cols).float().log_softmax(-1), lens  # fp32: CTC is unstable in bf16

    def loss(self, images, widths, targets, target_lens):
        """targets: (B, L) cluster ids without <s>/</s>, PAD after target_lens. Returns total, ctc, ce."""
        memory, mem_pad, ctc_lp, ctc_lens = self.encode(images, widths)
        ctc = F.ctc_loss(ctc_lp.transpose(0, 1), targets, ctc_lens, target_lens,
                         blank=BLANK, reduction="mean", zero_infinity=True)
        B = targets.shape[0]
        bos = torch.full((B, 1), BOS, device=targets.device)
        dec_in = torch.cat([bos, targets], 1)
        dec_out = torch.cat([targets, torch.full((B, 1), PAD, device=targets.device)], 1)
        dec_out[torch.arange(B), target_lens] = EOS
        logits = self.decoder(dec_in, memory, mem_pad).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), dec_out.reshape(-1), ignore_index=PAD,
                             label_smoothing=self.c.smooth)
        w = self.c.ctc_weight
        return w * ctc + (1 - w) * ce, ctc, ce

    def lm_loss(self, targets, target_lens):
        """Text-only pre-training of the decoder (no picture): learn which Khmer clusters follow each other."""
        B = targets.shape[0]
        dec_in = torch.cat([torch.full((B, 1), BOS, device=targets.device), targets], 1)
        dec_out = torch.cat([targets, torch.full((B, 1), PAD, device=targets.device)], 1)
        dec_out[torch.arange(B), target_lens] = EOS
        logits = self.decoder(dec_in)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), dec_out.reshape(-1), ignore_index=PAD)

    # ---- reading ------------------------------------------------------------------------------
    @staticmethod
    def ctc_greedy(ctc_lp, lens):
        """Best cluster per column, repeats merged, blanks removed."""
        best = ctc_lp.argmax(-1)
        out = []
        for b in range(best.shape[0]):
            seq, prev = [], None
            for t in best[b, : lens[b]].tolist():
                if t != prev and t != BLANK:
                    seq.append(t)
                prev = t
            out.append(seq)
        return out

    @torch.no_grad()
    def read(self, images, widths):
        """Greedy decoder with the RUNAWAY GUARD: the decoder may write at most 1.5 × (CTC count) + 4 clusters;
        if it goes past that, or never stops, the CTC draft is used instead. Returns (ids, used_ctc) per line."""
        memory, mem_pad, ctc_lp, lens = self.encode(images, widths)
        drafts = self.ctc_greedy(ctc_lp, lens)
        B = images.shape[0]
        limit = torch.tensor([int(1.5 * len(d)) + 4 for d in drafts], device=images.device)
        ids = torch.full((B, 1), BOS, device=images.device)
        done = torch.zeros(B, dtype=torch.bool, device=images.device)
        for _ in range(int(limit.max())):
            nxt = self.decoder(ids, memory, mem_pad)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ids = torch.cat([ids, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        out = []
        for b in range(B):
            seq = ids[b, 1:].tolist()
            if EOS in seq:
                out.append((seq[: seq.index(EOS)], False))
            else:   # ran away (no </s> within the limit): trust the CTC draft
                out.append((drafts[b], True))
        return out


def count_params(m: nn.Module) -> dict:
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    return {"cnn": n(m.cnn), "encoder": n(m.encoder), "ctc_head": n(m.ctc), "decoder": n(m.decoder), "total": n(m)}
