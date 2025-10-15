# app/utils_topk.py — Top-K chunking (FR), compatible tokenizers HF (mBART, etc.)
from typing import List, Tuple, Dict, Optional

# (facultatif) Mots-clés par focus pour biaiser le scoring, FR-centric
FOCUS_KEYWORDS: Dict[str, List[str]] = {
    "méthodes": ["méthode", "method", "approche", "algorithm", "algorithme", "training", "architecture", "procédure"],
    "détails": ["données", "dataset", "feature", "prétraitement", "hyperparam", "annotation", "implémentation"],
    "applications": ["application", "use case", "cas d'usage", "déploiement", "industrie", "production"],
    "résultats": ["résultat", "accuracy", "f1", "score", "benchmark", "table", "figure", "expérimental"],
    "limites": ["limite", "limitation", "future work", "contrainte", "biais", "risque", "erreur"],
    # "général": []  # pas de biais
}

def chunk_text_with_context(
    text: str,
    tokenizer,
    max_tokens: int = 900,
    overlap: int = 150,
) -> Tuple[List[List[int]], List[str]]:
    """
    Coupe `text` en fenêtres de tokens (max_tokens) avec recouvrement (overlap).
    Retourne (chunks_ids, chunks_txt).
    """
    if not text or not text.strip():
        return [], []

    enc = tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_tensors=None)
    ids = enc["input_ids"]
    # certains tokenizers renvoient [[...]] → aplatir
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = [t for seq in ids for t in seq]

    if max_tokens <= 0:
        raise ValueError("max_tokens doit être > 0")
    if overlap < 0 or overlap >= max_tokens:
        overlap = max(0, max_tokens // 6)  # garde-fou

    step = max_tokens - overlap
    chunks_ids: List[List[int]] = []
    for start in range(0, len(ids), step):
        window = ids[start:start + max_tokens]
        if not window:
            break
        chunks_ids.append(window)

    chunks_txt: List[str] = [tokenizer.decode(c, skip_special_tokens=True) for c in chunks_ids]
    return chunks_ids, chunks_txt


def score_chunk_fr(text: str, focus: Optional[str]) -> float:
    """
    score = 0.55 * biais_focus + 0.30 * densité de chiffres (bornée) + 0.15 * diversité lexicale
    focus peut être None → pas de biais focus.
    """
    t = (text or "").lower().strip()
    if not t:
        return 0.0

    if focus:
        kws = FOCUS_KEYWORDS.get(focus, [])
        kw_hits = sum(1 for k in kws if k in t)
        kw_part = 0.55 * (kw_hits / max(1, len(kws))) if kws else 0.0
    else:
        kw_part = 0.0

    digits = sum(c.isdigit() for c in t)  # ✅ correction parenthèses
    num_part = 0.30 * min(digits / 50.0, 1.0)

    toks = t.split()
    uniq_ratio = len(set(toks)) / max(1, len(toks))
    div_part = 0.15 * uniq_ratio

    return float(kw_part + num_part + div_part)


def select_topk_chunks(
    text: str,
    tokenizer,
    *,
    k: int = 1,
    focus: Optional[str] = None,
    prefix: str = "",
    ctx_max: int = 1024,
    safety_ratio: float = 0.8,
    max_tokens: int = 900,
    overlap: int = 150,
    enable_topk: bool = True,
) -> Dict[str, object]:
    """
    Chunk -> score -> tri -> sélection Top-K (+ fast-path si 1 chunk tient avec le prefix)
    Retour:
      {
        'chunks_ids': List[List[int]],
        'chunks_txt': List[str],
        'selected_idx': List[int],
        'selected_txt': List[str],
        'fast_path': bool,
        'prefix_tokens': int,
        'avg_chunk_tokens': int
      }
    """
    chunks_ids, chunks_txt = chunk_text_with_context(text, tokenizer, max_tokens=max_tokens, overlap=overlap)
    if not chunks_txt:
        return dict(chunks_ids=[], chunks_txt=[], selected_idx=[], selected_txt=[], fast_path=False,
                    prefix_tokens=0, avg_chunk_tokens=0)

    prefix_tokens = len(tokenizer(prefix, add_special_tokens=False).input_ids) if prefix else 0
    avg_chunk_tokens = int(
        sum(len(tokenizer(c, add_special_tokens=False).input_ids) for c in chunks_txt) / max(1, len(chunks_txt))
    )

    fits_single = (prefix_tokens + avg_chunk_tokens) < int(safety_ratio * ctx_max)

    if not enable_topk:
        sel = list(range(len(chunks_txt)))
        fast_path = False
    else:
        scored = [(i, score_chunk_fr(chunks_txt[i], focus)) for i in range(len(chunks_txt))]
        scored.sort(key=lambda x: x[1], reverse=True)
        kk = max(1, min(k, len(scored)))
        if kk == 1 and fits_single:
            sel = [scored[0][0]]
            fast_path = True
        else:
            sel = [i for (i, _) in scored[:kk]]
            fast_path = False

    selected_txt = [chunks_txt[i] for i in sel]
    return dict(
        chunks_ids=chunks_ids,
        chunks_txt=chunks_txt,
        selected_idx=sel,
        selected_txt=selected_txt,
        fast_path=fast_path,
        prefix_tokens=int(prefix_tokens),
        avg_chunk_tokens=int(avg_chunk_tokens),
    )
