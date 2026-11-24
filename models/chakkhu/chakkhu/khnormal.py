"""Khmer text normalization: two correct transcriptions of the same picture must become the SAME string.

Unicode NFC does almost nothing for Khmer (nearly all marks have combining class 0), so the same visible
cluster can be typed in several orders. We put every cluster into one fixed order and fix common lookalikes:

  1. invisible characters removed: ZWSP U+200B, ZWNJ U+200C, ZWJ U+200D, soft hyphen U+00AD, BOM;
     deprecated invisible vowels U+17B4 / U+17B5
  2. deprecated letters: ឣ (U+17A3) → អ,  ឤ (U+17A4) → អា
  3. split vowels joined:  េ + ា → ោ,  េ + ី → ើ,  េ + ៀ → ៀ  (typed in two pieces, look the same)
  4. one order inside a cluster:
         base · subscripts (្ + consonant; ្រ last) · shifter ៉ ៊ · robat ៌ · vowel · signs (ំ ះ ់ ៍ …)
     (ាំ is vowel + sign, so it always comes out as ា then ំ)
  5. doubled marks removed: ្្ → ្, the same vowel/sign typed twice in one cluster → once

Words that LOOK different stay different (ឲ vs ឱ្យ, ដ្ត vs ដ្ដ): the picture tells them apart.
"""
from __future__ import annotations

import re
import unicodedata

COENG, RO = "្", "រ"
SHIFTERS = "៉៊"                     # ៉ ៊
ROBAT = "៌"                              # ៌
VOWELS = set(chr(c) for c in range(0x17B6, 0x17C6))   # ា … ៅ
SIGNS = set("ំះៈ់៍៎៏័៑៓៝")  # ំ ះ ៈ ់ ៍ ៎ ៏ ័ ៑ ៓ ៝
CONS = set(chr(c) for c in range(0x1780, 0x17A3))     # consonants ក … អ
BASES = set(chr(c) for c in range(0x1780, 0x17B4))    # consonants + independent vowels

INVISIBLE = dict.fromkeys(map(ord, "​‌‍­﻿឴឵"), None)
REPLACE = [("ឣ", "អ"), ("ឤ", "អា"),
           ("េា", "ោ"), ("េី", "ើ"), ("េៀ", "ៀ")]

# a cluster: a base, then any marks (subscripts written as ្ + consonant)
CLUSTER = re.compile(r"[ក-ឳ](?:្[ក-ឳ]|[ា-៑៓៝])*")


def _order(cluster: str) -> str:
    base, rest = cluster[0], cluster[1:]
    subs, shift, robat, vowels, signs = [], [], [], [], []
    i = 0
    while i < len(rest):
        c = rest[i]
        if c == COENG:
            if i + 1 < len(rest) and rest[i + 1] in BASES:
                subs.append(COENG + rest[i + 1])
                i += 2
                continue
            i += 1                       # a lone or doubled ្: dropped
            continue
        if c in SHIFTERS:
            shift.append(c)
        elif c == ROBAT:
            robat.append(c)
        elif c in VOWELS:
            vowels.append(c)
        else:
            signs.append(c)
        i += 1
    # ្រ is written last among the subscripts (it is drawn on the left, around the others)
    subs = [s for s in subs if s[1] != RO] + [s for s in subs if s[1] == RO]

    def once(xs):  # the same mark typed twice → once (keeps the first order)
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    return base + "".join(subs) + "".join(once(shift)) + "".join(once(robat)) + "".join(once(vowels)) + \
        "".join(once(signs))


def khnormal(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(INVISIBLE)
    for a, b in REPLACE:
        text = text.replace(a, b)
    text = re.sub("្{2,}", COENG, text)   # ្្ → ្
    text = CLUSTER.sub(lambda m: _order(m.group(0)), text)
    for a, b in REPLACE[2:]:             # vowels can meet again after reordering
        text = text.replace(a, b)
    return text
