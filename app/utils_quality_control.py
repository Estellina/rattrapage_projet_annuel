# app/utils_quality.py
from __future__ import annotations
import re
from typing import Tuple

def _distinct_n(text: str, n: int = 2) -> float:
    t = " ".join(text.lower().split())
    toks = t.split()
    if len(toks) < n: return 1.0
    ngrams = set(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
    return len(ngrams) / max(1, len(toks)-n+1)

def _len_ratio(src: str, out: str, lo: float = 0.10, hi: float = 0.85) -> float:
    if not src or not out: return 0.0
    r = len(out) / max(1, len(src))
    if r < lo: return r/lo
    if r > hi: return max(0.0, 1.0 - (r-hi)/(2*hi))
    return 1.0

def _overlap(src: str, out: str) -> float:
    if not src or not out: return 0.0
    S = set(src.lower().split())
    O = set(out.lower().split())
    if not S: return 0.0
    return len(S & O) / max(1, len(S))

def simple_quality_score(src_text: str, summary_text: str) -> float:
    """
    Score [0..1] combinant:
      - couverture (overlap lexical),
      - concision (ratio de longueur),
      - diversité (distinct-2 / distinct-3),
      - propreté (ponctuation de base).
    C’est un signal diag, pas un “metric officiel”.
    """
    src = (src_text or "").strip()
    out = (summary_text or "").strip()
    if not out: return 0.0

    cov = _overlap(src, out)
    ratio = _len_ratio(src, out)
    d2 = _distinct_n(out, 2)
    d3 = _distinct_n(out, 3)
    punct = 1.0 - min(1.0, len(re.findall(r"[!?.,;:]{3,}", out)) / 3.0)

    # pondérations pragmatiques
    score = 0.35*cov + 0.25*ratio + 0.20*d2 + 0.10*d3 + 0.10*punct
    return round(max(0.0, min(1.0, score)), 3)
