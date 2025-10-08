"""
Loader and inference routines for a custom question generation (QG) model.

This module provides a minimal implementation for loading a transformer‐based
question generator trained from scratch and performing biased beam search
decoding.  The implementation is derived from the provided notebook
``questions_models_custom.ipynb``.  It defines a ``QGTransformer`` class
matching the training architecture, loads a tokeniser from disk, and
implements a beam search with keyword biasing, reranking and de‑duplication.

Functions exported:

    * ``load_qg_model(model_path, tokenizer_dir)`` – load the QG model and
      associated tokeniser.  After calling this, ``generate_questions`` can
      be used to perform inference.

    * ``generate_questions(text_en, n, difficulty, scope, section)`` –
      generate ``n`` English questions from the provided text.  Difficulty,
      scope and section parameters are currently ignored but retained for
      API compatibility.

If your own model differs substantially from this implementation (e.g. you
use a different decoder or keyword biasing strategy), adapt the code
accordingly.  The default logic will always attempt to produce at least
``n`` questions, reusing text keywords to ensure relevance.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import List, Tuple, Set, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Tokenisers: use HuggingFace's tokenizers library if available.  We
    # support both a JSON file (tokenizer.json) and the pair of vocab/merges
    # used by ByteLevelBPETokenizer.  The Tokenizer class is used for QG.
    from tokenizers import Tokenizer, decoders
    from tokenizers.implementations import ByteLevelBPETokenizer
except ImportError as exc:  # pragma: no cover – handled at runtime
    Tokenizer = None  # type: ignore
    ByteLevelBPETokenizer = None  # type: ignore
    decoders = None  # type: ignore
    raise


###############################################################################
#  Model definition
###############################################################################

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encodings with dropout.

    The implementation follows the notebook and transforms an input of shape
    ``(seq_len, batch, d_model)`` by adding a fixed positional vector and
    applying dropout.  We transpose inputs as necessary in the QG model.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch, d_model]
        seq_len = x.size(0)
        x = x + self.pe[:seq_len].unsqueeze(1)
        return self.dropout(x)


class QGTransformer(nn.Module):
    """Encoder–decoder transformer for question generation.

    This model mirrors the architecture used in the training notebook: a
    standard Transformer encoder and decoder with shared embedding sizes
    and positional encodings.  The number of layers and hidden size
    default to the values used during training, but can be adjusted if
    your checkpoint differs.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_enc: int = 6,
        num_dec: int = 6,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=False)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _src_key_padding_mask(self, src_ids: torch.Tensor) -> torch.Tensor:
        # src_ids: [B,S]
        return src_ids == self.pad_id

    def _tgt_masks(self, tgt_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # tgt_ids: [B,T]
        B, T = tgt_ids.size()
        causal = torch.triu(torch.ones(T, T, device=tgt_ids.device), diagonal=1).bool()
        pad_mask = tgt_ids == self.pad_id
        return causal, pad_mask

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        # src_ids: [B,S], tgt_ids: [B,T]
        src_mask = self._src_key_padding_mask(src_ids)  # [B,S]
        tgt_mask, tgt_key_padding = self._tgt_masks(tgt_ids)  # [T,T], [B,T]
        src = self.src_embed(src_ids).transpose(0, 1)  # [S,B,E]
        tgt = self.tgt_embed(tgt_ids).transpose(0, 1)  # [T,B,E]
        src = self.pos_enc(src)
        tgt = self.pos_enc(tgt)
        mem = self.encoder(src, src_key_padding_mask=src_mask)
        out = self.decoder(tgt, mem, tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_key_padding,
                           memory_key_padding_mask=src_mask)
        logits = self.lm_head(out).transpose(0, 1)  # [B,T,V]
        return logits


###############################################################################
#  Global state
###############################################################################

_model: Optional[QGTransformer] = None
_tokenizer: Optional[object] = None  # tokenizers.Tokenizer or ByteLevelBPETokenizer
PAD_ID: Optional[int] = None
BOS_ID: Optional[int] = None
EOS_ID: Optional[int] = None


###############################################################################
#  Tokeniser helpers
###############################################################################

def _load_tokenizer_from_dir(tokenizer_dir: Path):
    """Detect and load a tokenizer from a directory.

    Tries to load a ``tokenizer.json`` via ``Tokenizer.from_file``.  If that
    does not exist, falls back to a ByteLevel BPE tokenizer using
    ``vocab.json`` and ``merges.txt``.  Returns a tuple of (tokenizer, pad_id,
    bos_id, eos_id).  Raises FileNotFoundError if no suitable files are
    present.
    """
    json_path = tokenizer_dir / "tokenizer.json"
    if json_path.is_file() and Tokenizer is not None:
        tok = Tokenizer.from_file(str(json_path))
        # Ensure byte‑level decoding for readability
        if decoders is not None:
            try:
                tok.decoder = decoders.ByteLevel()
            except Exception:
                pass
        # Determine special tokens (fallback to common names)
        pad_id = tok.token_to_id("[PAD]") or tok.token_to_id("<pad>")
        bos_id = tok.token_to_id("[BOS]") or tok.token_to_id("<s>") or tok.token_to_id("<bos>")
        eos_id = tok.token_to_id("[EOS]") or tok.token_to_id("</s>") or tok.token_to_id("<eos>")
        if None in (pad_id, bos_id, eos_id):
            raise RuntimeError("Special tokens [PAD]/[BOS]/[EOS] missing from tokenizer")
        return tok, int(pad_id), int(bos_id), int(eos_id)
    # Fallback: vocab/merges for ByteLevel BPE
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"
    if vocab_path.is_file() and merges_path.is_file() and ByteLevelBPETokenizer is not None:
        tok = ByteLevelBPETokenizer.from_file(str(vocab_path), str(merges_path))
        pad_id = tok.token_to_id("<pad>")
        bos_id = tok.token_to_id("<s>")
        eos_id = tok.token_to_id("</s>")
        if None in (pad_id, bos_id, eos_id):
            raise RuntimeError("Special tokens <pad>/<s></s> missing from tokenizer")
        try:
            tok.special_tokens_map = {"pad_token": "<pad>", "bos_token": "<s>", "eos_token": "</s>"}
        except Exception:
            pass
        return tok, int(pad_id), int(bos_id), int(eos_id)
    raise FileNotFoundError(
        f"No supported tokenizer files found in {tokenizer_dir}. Expected tokenizer.json or vocab.json/merges.txt."
    )


###############################################################################
#  Beam search with keyword biasing
###############################################################################

def _encode_str(text: str, max_len: int) -> List[int]:
    """Encode a string using the loaded tokenizer and append EOS.

    Truncates to ``max_len - 1`` tokens and appends ``EOS_ID``.
    """
    assert _tokenizer is not None and EOS_ID is not None
    # tokenizers library returns an object with ``ids`` or a list
    enc = _tokenizer.encode(text)
    ids: List[int]
    if hasattr(enc, "ids"):
        ids = [int(i) for i in enc.ids]
    elif isinstance(enc, list):
        ids = [int(i) for i in enc]
    else:
        ids = [int(enc)]
    ids = ids[: max_len - 1] + [EOS_ID]
    return ids


STOP_WORDS: Set[str] = set(
    """
a an the and or of to in for on with at from by as is are was were be being been it this that which who whose whom
""".split()
)


def _extract_keywords(text: str, k: int = 40) -> List[str]:
    """Return up to ``k`` informative keywords from ``text``.

    Words shorter than four letters or present in a stop list are ignored.
    Frequencies are counted case‑insensitively and the top ``k`` words are
    returned.
    """
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text)
    words = [w.lower() for w in words if w.lower() not in STOP_WORDS]
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:k]]


def _keyword_token_ids(keywords: List[str]) -> Set[int]:
    """Convert a list of keywords into a set of token IDs using the tokenizer.

    Each keyword may map to multiple subword tokens.  Special tokens are
    excluded from the result.
    """
    ids: Set[int] = set()
    assert _tokenizer is not None and PAD_ID is not None and BOS_ID is not None and EOS_ID is not None
    for kw in keywords:
        enc = _tokenizer.encode(kw)
        if hasattr(enc, "ids"):
            sub_ids = [int(i) for i in enc.ids]
        elif isinstance(enc, list):
            sub_ids = [int(i) for i in enc]
        else:
            sub_ids = [int(enc)]
        for tid in sub_ids:
            if tid not in (PAD_ID, BOS_ID, EOS_ID):
                ids.add(tid)
    return ids


def _postprocess_questions(cands: List[str], context_text: str) -> List[str]:
    """Clean up, deduplicate and rerank a list of candidate questions.

    Performs the following operations:
        * strip whitespace and remove numeric prefixes (e.g. "1. ").
        * ensure each question ends with a question mark.
        * deduplicate questions ignoring punctuation and case.
        * rerank questions based on their lexical overlap with the context,
          penalising overly trivial openers and extreme lengths.
    Returns:
        A list of cleaned and ranked questions.
    """
    out: List[str] = []
    for q in cands:
        q = q.strip()
        q = re.sub(r"^\s*\d+\s*[\.\)\:-]\s*", "", q)
        q = re.sub(r"^(Question|Q:)\s*", "", q, flags=re.I)
        if not q.endswith("?"):
            q = re.sub(r"[.]+$", "", q).strip() + "?"
        if q:
            out.append(q[0].upper() + q[1:])
    # Deduplicate
    seen: Set[str] = set()
    uniq: List[str] = []
    for q in out:
        key = re.sub(r"\W+", "", q.lower())
        if key not in seen:
            seen.add(key)
            uniq.append(q)
    ctx = context_text.lower()
    def score(q: str) -> float:
        ql = q.lower()
        toks = re.findall(r"[a-z][a-z\-]{3,}", ql)
        ov = sum(1 for w in toks if w in ctx)
        # trivial penalty
        trivial = bool(re.match(r"^(what\s+is|what\s+are|define)\b", ql))
        pen_triv = 0.0 if not trivial else 0.25
        # length penalty
        L = len(q.split())
        len_pen = 0.0
        if L < 8:
            len_pen += 0.05 * (8 - L)
        if L > 28:
            len_pen += 0.03 * (L - 28)
        return ov - pen_triv - len_pen
    uniq.sort(key=score, reverse=True)
    return uniq


@torch.no_grad()
def _beam_search_generate_batch_biased(
    src_ids: torch.Tensor,
    context_text: str,
    num_beams: int = 6,
    ret_n: int = 16,
    max_new_tokens: int = 64,
    repetition_penalty: float = 1.15,
    kw_bonus: float = 0.8,
) -> List[List[int]]:
    """Perform beam search decoding with keyword biasing.

    Args:
        src_ids: A tensor of shape ``(1, S)`` containing encoded source tokens.
        context_text: The raw text used to extract keywords for biasing.
        num_beams: Beam width.
        ret_n: Number of sequences to return.
        max_new_tokens: Maximum number of tokens to generate (excluding BOS/EOS).
        repetition_penalty: Factor by which to penalise tokens already in the
            sequence.  Values >1 discourage repetition.
        kw_bonus: Additive logit bonus applied to keyword tokens.
    Returns:
        A list of generated token sequences (excluding BOS/EOS) sorted by
        approximate probability.
    """
    assert _model is not None and _tokenizer is not None and PAD_ID is not None and BOS_ID is not None and EOS_ID is not None
    device = next(_model.parameters()).device
    assert src_ids.dim() == 2 and src_ids.size(0) == 1
    # Precompute keyword IDs
    kws = _extract_keywords(context_text, k=50)
    kwids = _keyword_token_ids(kws)
    # Each beam is (tensor[seq], cumulative logprob)
    beams: List[Tuple[torch.Tensor, float]] = [(torch.tensor([BOS_ID], device=device), 0.0)]
    for _ in range(max_new_tokens):
        # Prepare batch of target sequences and replicate source
        tgt_batch = torch.nn.utils.rnn.pad_sequence(
            [seq for seq, _ in beams], batch_first=True, padding_value=PAD_ID
        )
        src_batch = src_ids.expand(tgt_batch.size(0), -1)
        logits = _model(src_batch, tgt_batch)  # [B,T,V]
        last = logits[:, -1, :]  # [B,V]
        # Repetition penalty: divide logits for tokens present in the sequence
        if repetition_penalty and repetition_penalty != 1.0:
            V = last.size(1)
            for i, (seq, _) in enumerate(beams):
                for token_id in set(seq.tolist()):
                    if 0 <= token_id < V:
                        last[i, token_id] /= repetition_penalty
        # Keyword bonus: boost logits of keyword IDs
        if kw_bonus and kwids:
            V = last.size(1)
            for tid in kwids:
                if 0 <= tid < V:
                    last[:, tid] += kw_bonus
        logprobs = torch.log_softmax(last, dim=-1)
        topk_vals, topk_idx = torch.topk(logprobs, k=num_beams, dim=-1)
        new_beams: List[Tuple[torch.Tensor, float]] = []
        all_eos = True
        for (seq, score), vals, idxs in zip(beams, topk_vals, topk_idx):
            if seq[-1].item() == EOS_ID:
                new_beams.append((seq, score))
                continue
            all_eos = False
            for k in range(num_beams):
                new_seq = torch.cat([seq, idxs[k].view(1)])
                new_score = score + vals[k].item()
                new_beams.append((new_seq, new_score))
        if all_eos:
            beams = new_beams
            break
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[: num_beams]
    # Sort final beams and strip special tokens
    beams.sort(key=lambda x: x[1], reverse=True)
    seqs: List[List[int]] = []
    for seq, _ in beams[:ret_n]:
        toks = seq.tolist()
        # Remove BOS and EOS if present
        if toks and toks[0] == BOS_ID:
            toks = toks[1:]
        if toks and toks[-1] == EOS_ID:
            toks = toks[:-1]
        seqs.append(toks)
    return seqs


@torch.no_grad()
def _generate_min_questions(
    text_en: str,
    min_n: int = 5,
    max_src_len: int = 512,
    max_new_tokens: int = 72,
) -> List[str]:
    """Generate at least ``min_n`` candidate questions from a passage.

    Uses a two‑stage biased beam search: first with moderate keyword bias to
    produce several candidates, then again with stronger bias if not enough
    candidates are produced.  The questions are postprocessed, deduplicated
    and reranked.
    Args:
        text_en: The English passage to base questions on.  Typically a
            summary of a PDF section.
        min_n: Minimum number of questions to return.
        max_src_len: Maximum length of the encoded source (in tokens).
        max_new_tokens: Maximum length of the generated sequence (in tokens).
    Returns:
        A list of questions (strings) sorted by relevance.
    """
    # Encode source with prefix "qg: " as done in training
    prefix = "qg: "
    ids = _encode_str(prefix + text_en.strip(), max_src_len)
    src_ids = torch.tensor([ids], dtype=torch.long, device=next(_model.parameters()).device)
    seqs = _beam_search_generate_batch_biased(
        src_ids,
        context_text=text_en,
        num_beams=6,
        ret_n=24,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.2,
        kw_bonus=0.9,
    )
    texts = [_tokenizer.decode(s) if hasattr(_tokenizer, "decode") else "".join(map(str, s)) for s in seqs]
    ranked = _postprocess_questions(texts, context_text=text_en)
    if len(ranked) < min_n:
        seqs2 = _beam_search_generate_batch_biased(
            src_ids,
            context_text=text_en,
            num_beams=6,
            ret_n=24,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.3,
            kw_bonus=1.2,
        )
        texts2 = [_tokenizer.decode(s) if hasattr(_tokenizer, "decode") else "".join(map(str, s)) for s in seqs2]
        ranked = _postprocess_questions(ranked + texts2, context_text=text_en)
    return ranked[:min_n]


###############################################################################
#  Public API
###############################################################################

def load_qg_model(model_path: str, tokenizer_dir: str) -> None:
    """Load the question generation model and tokenizer from disk.

    Args:
        model_path: Path to the ``.pth`` checkpoint containing the model
            weights.  The checkpoint should either be a plain state dict
            or contain a key ``"model"`` with the state dict.
        tokenizer_dir: Path to the directory containing the tokenizer files
            (``tokenizer.json`` or ``vocab.json`` and ``merges.txt``).
    Raises:
        FileNotFoundError: If no tokenizer files are found.
        RuntimeError: If the tokenizer or model cannot be loaded.
    """
    global _model, _tokenizer, PAD_ID, BOS_ID, EOS_ID
    model_p = Path(model_path)
    tok_dir = Path(tokenizer_dir)
    if not model_p.is_file():
        raise FileNotFoundError(f"QG model checkpoint not found: {model_path}")
    # Load tokenizer and special IDs
    tok, pad_id, bos_id, eos_id = _load_tokenizer_from_dir(tok_dir)
    PAD_ID, BOS_ID, EOS_ID = pad_id, bos_id, eos_id
    _tokenizer = tok
    vocab_size = tok.get_vocab_size() if hasattr(tok, "get_vocab_size") else len(tok.get_vocab())
    # Load state dict
    state = torch.load(model_p, map_location="cpu")
    if isinstance(state, dict) and any(isinstance(v, torch.Tensor) for v in state.values()):
        sd = state
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        raise RuntimeError(f"Unexpected checkpoint format for {model_path}")
    # Instantiate model.  Adjust hyper‑parameters here if your model differs.
    model = QGTransformer(vocab_size=vocab_size, pad_id=pad_id)
    model.load_state_dict(sd)
    model.eval()
    _model = model


def generate_questions(
    text_en: str,
    n: int = 5,
    difficulty: Optional[str] = None,
    scope: Optional[str] = None,
    section: Optional[str] = None,
) -> List[str]:
    """Generate a list of ``n`` questions about a given English text.

    This function wraps ``_generate_min_questions`` and truncates the result
    to the desired number of questions.  The ``difficulty``, ``scope`` and
    ``section`` arguments are ignored in this implementation but retained
    for interface compatibility with the API.  Ensure that
    ``load_qg_model`` has been called before invoking this function.

    Args:
        text_en: The English passage from which to generate questions.
        n: Desired number of questions to return.
        difficulty: Unused (for compatibility).
        scope: Unused (for compatibility).
        section: Unused (for compatibility).
    Returns:
        A list of questions (strings).
    Raises:
        RuntimeError: If the QG model has not been loaded.
    """
    if _model is None or _tokenizer is None or PAD_ID is None or BOS_ID is None or EOS_ID is None:
        raise RuntimeError(
            "QG model not loaded. Call load_qg_model(model_path, tokenizer_dir) first."
        )
    # Generate at least n questions; then slice
    qs = _generate_min_questions(text_en, min_n=max(1, n))
    return qs[:n]