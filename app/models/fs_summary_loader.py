"""
Loader and inference routines for a custom hierarchical summarization model.

This module exposes two top‑level functions:

    * ``load_summary_model(model_path, tokenizer_dir)`` – load a Byte‑Level BPE tokenizer
      from ``tokenizer_dir`` and a HIBERT‑like model from ``model_path``.  After
      calling this function the global variables ``_model`` and ``_tokenizer``
      point to the loaded objects and can be used for inference.

    * ``generate_summary(text_en)`` – given an English text, produce a concise
      summary using the loaded model.  The input is automatically split into
      sentences, tokenised, packed into the hierarchical tensor format and
      decoded with a greedy decoder biased towards the vocabulary of the
      source.  See the notebook ``scientific_training (3).ipynb`` for the
      underlying architecture and generation strategy.

When integrating your own from‑scratch summariser you should replace or
extend the implementation below.  The current code mirrors the logic
found in the provided notebook: it re‑implements the ``HIBERTLike`` class,
loads a ByteLevelBPETokenizer from disk and performs grounded greedy
decoding.  If your checkpoint contains different hyper‑parameters you may
need to adjust the arguments passed to ``HIBERTLike`` to match those used
during training.
"""

from __future__ import annotations

import os
import math
import re
from pathlib import Path
from typing import Optional, Tuple, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # The tokenizers library provides a fast ByteLevelBPETokenizer used during
    # training.  If it is not installed the loader will fail.  Install via
    # ``pip install tokenizers``.
    from tokenizers.implementations import ByteLevelBPETokenizer
except ImportError as exc:  # pragma: no cover – handled at runtime
    ByteLevelBPETokenizer = None
    raise


###############################################################################
#  HIBERT‑like architecture
###############################################################################

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encodings with dropout.

    This module is reused for tokens, sentences and decoder inputs.  It
    constructs a ``(max_len, d_model)`` buffer and adds it to the input
    embeddings.  The implementation is identical to that used in the
    reference notebook.
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape to [1, max_len, d_model] so it can be added to batch inputs
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``x`` and apply dropout.

        Args:
            x: Tensor of shape ``(batch_size, seq_len, d_model)``.
        Returns:
            Tensor of the same shape as ``x``.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class HIBERTLike(nn.Module):
    """Hierarchical encoder–decoder for long document summarisation.

    The model encodes sentences of tokens with one transformer encoder,
    aggregates them by mean‑pooling into sentence embeddings, encodes those
    embeddings with a second transformer and decodes a summary with a
    standard Transformer decoder.  The default hyper‑parameters mirror
    those used in the notebook.  If your checkpoint was trained with
    different dimensions you may need to modify the defaults accordingly.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 1024,
        nhead: int = 16,
        num_enc1_layers: int = 4,
        num_enc2_layers: int = 4,
        num_dec_layers: int = 8,
        dim_ff: int = 4096,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id

        # Embeddings and positional encodings
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_tok = PositionalEncoding(d_model, dropout=dropout)
        self.pos_sent = PositionalEncoding(d_model, dropout=dropout)

        # Token‑level encoder
        enc1_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.enc1 = nn.TransformerEncoder(enc1_layer, num_layers=num_enc1_layers)

        # Sentence‑level encoder
        enc2_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.enc2 = nn.TransformerEncoder(enc2_layer, num_layers=num_enc2_layers)

        # Decoder
        self.dec_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_dec = PositionalEncoding(d_model, dropout=dropout)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)
        self.out = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(L: int, device: torch.device) -> torch.Tensor:
        """Return an upper triangular matrix of ``True`` values, masking the future."""
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        src: torch.Tensor,
        src_mask_tokens: torch.Tensor,
        src_mask_sents: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training and greedy decoding.

        Args:
            src: Tensor of shape ``(B, S, T)`` with token IDs.
            src_mask_tokens: Boolean tensor of shape ``(B, S, T)`` where ``True``
                indicates a valid token (not padding).
            src_mask_sents: Boolean tensor of shape ``(B, S)`` where ``True``
                indicates a valid sentence (not padding).
            tgt: Tensor of shape ``(B, L)`` containing decoder input IDs.  During
                training this should be ``tgt_in = tgt[:, :-1]``.  During greedy
                decoding this function is invoked step‑by‑step.
        Returns:
            Logits of shape ``(B, L, vocab_size)``.
        """
        B, S, T = src.size()
        device = src.device

        # --- Encoder 1: token‑level ---
        x = self.tok_emb(src.reshape(B * S, T))  # [B*S,T,D]
        x = self.pos_tok(x)
        key_padding = ~src_mask_tokens.reshape(B * S, T)  # True=pad
        h = self.enc1(x, src_key_padding_mask=key_padding)  # [B*S,T,D]

        # Mean‑pool per sentence (masked)
        mask = src_mask_tokens.reshape(B * S, T).unsqueeze(-1)
        h_sum = (h * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-5)
        sent_emb = (h_sum / lengths).reshape(B, S, -1)  # [B,S,D]

        # --- Encoder 2: sentence‑level ---
        sent_pos = self.pos_sent(sent_emb)
        sent_key_padding = ~src_mask_sents  # True=pad
        memory = self.enc2(sent_pos, src_key_padding_mask=sent_key_padding)  # [B,S,D]

        # --- Decoder ---
        tgt_emb = self.pos_dec(self.dec_emb(tgt))  # [B,L,D]
        causal = self._causal_mask(tgt.size(1), device)  # [L,L]
        tgt_pad = (tgt == self.pad_id)
        mem_pad = ~src_mask_sents
        out = self.dec(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=mem_pad,
        )  # -> [B,L,D]
        logits = self.out(out)
        return logits


###############################################################################
#  Tokeniser and packing helpers
###############################################################################

# Global variables holding the loaded model and tokenizer.  They are set when
# ``load_summary_model`` is called and used by ``generate_summary``.
_model: Optional[HIBERTLike] = None
_tokenizer: Optional[ByteLevelBPETokenizer] = None

# IDs for special tokens are initialised after loading the tokenizer
PAD: Optional[int] = None
BOS: Optional[int] = None
EOS: Optional[int] = None


def _safe_encode(tokenizer: ByteLevelBPETokenizer, text_or_token: str) -> list[int]:
    """Encode a piece of text robustly with the provided tokenizer.

    Returns a list of integer IDs.  This helper gracefully handles cases
    where the underlying tokeniser returns a custom object rather than a
    raw list and catches any exceptions by returning an empty list.
    """
    try:
        enc = tokenizer.encode(text_or_token)
        # tokenizers library returns an object with ``ids``
        if hasattr(enc, "ids"):
            return [int(i) for i in enc.ids]
        # fallback: treat as list
        elif isinstance(enc, list):
            return [int(i) for i in enc]
        else:
            return [int(enc)]
    except Exception:
        return []


def _extract_src_vocab_from_tensor(
    src: torch.Tensor,
    sm_t: Optional[torch.Tensor],
    special_ids: Set[int],
) -> Set[int]:
    """Extract the set of visible token IDs from a packed source tensor.

    Args:
        src: Tensor of shape ``(B, S, L)`` containing token IDs.
        sm_t: Optional boolean mask of shape ``(B, S, L)`` where ``True`` denotes
            a valid token.  If ``None`` then all positions are treated as
            valid.
        special_ids: Set of integer IDs to exclude (e.g. PAD/BOS/EOS).
    Returns:
        A Python set of integer IDs present in the source (excluding
        ``special_ids``).  Only works for ``B==1`` as used during inference.
    """
    if src.size(0) != 1:
        raise ValueError("_extract_src_vocab_from_tensor expects a batch size of 1")
    if sm_t is not None:
        vis = src.masked_select(sm_t.to(src.device))
    else:
        vis = src.view(-1)
    ids = [int(i) for i in vis.tolist() if int(i) not in special_ids]
    return set(ids)


def _build_bias_mask(
    vocab_size: int,
    allowed_ids: Set[int],
    device: torch.device,
    bias_allowed: float = 0.0,
    bias_block: float = -6.0,
) -> torch.Tensor:
    """Create a bias vector to encourage or discourage certain tokens.

    The returned tensor has shape ``(vocab_size,)`` and contains
    ``bias_block`` for all positions except those in ``allowed_ids`` which
    receive ``bias_allowed`` (usually zero).  Adding this tensor to the
    decoder logits pushes the model to favour or penalise certain tokens.
    """
    bias = torch.full((vocab_size,), bias_block, dtype=torch.float32, device=device)
    if allowed_ids:
        idx = torch.tensor(sorted(list(allowed_ids)), dtype=torch.long, device=device)
        bias.index_fill_(0, idx, bias_allowed)
    return bias


def pack_from_raw_text(
    raw_text: str,
    tokenizer: ByteLevelBPETokenizer,
    pad_id: int,
    max_sents: int = 48,
    max_tok_per_sent: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack raw text into hierarchical tensors expected by ``HIBERTLike``.

    Splits the raw text into sentences using a simple regex on punctuation
    and newlines, then tokenises each sentence and truncates to
    ``max_tok_per_sent`` tokens.  Returns the packed ``src`` tensor and
    boolean masks indicating which token positions and sentences are valid.
    Args:
        raw_text: The input text (English) to be summarised.
        tokenizer: A ByteLevelBPETokenizer with special tokens defined.
        pad_id: The integer ID used for padding tokens.
        max_sents: Maximum number of sentences to include.
        max_tok_per_sent: Maximum number of tokens per sentence.
    Returns:
        src: Long tensor of shape ``(1, S, T)`` with token IDs.
        sm_t: Boolean tensor ``(1, S, T)`` where ``True`` marks a valid token.
        sm_s: Boolean tensor ``(1, S)`` where ``True`` marks a valid sentence.
    """
    # Basic sentence splitter: split on .?! followed by whitespace or newline
    SPLIT = re.compile(r"(?<=[\.\?\!])\s+|\n+")
    sents = [s.strip() for s in SPLIT.split(raw_text) if s.strip()]
    sents = sents[: max_sents]
    enc: list[list[int]] = []
    for s in sents:
        ids = _safe_encode(tokenizer, s)[: max_tok_per_sent]
        enc.append(ids if ids else [pad_id])
    S = len(enc)
    L = max(len(x) for x in enc) if S > 0 else 1
    # Pack into tensors of shape (1, S, L)
    src = torch.full((1, S, L), pad_id, dtype=torch.long)
    sm_t = torch.zeros((1, S, L), dtype=torch.bool)
    for i, ids in enumerate(enc):
        src[0, i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        sm_t[0, i, : len(ids)] = True
    sm_s = torch.ones((1, S), dtype=torch.bool)
    return src, sm_t, sm_s


@torch.no_grad()
def generate_grounded(
    model: HIBERTLike,
    src: torch.Tensor,
    sm_t: Optional[torch.Tensor],
    sm_s: Optional[torch.Tensor],
    BOS_id: int,
    EOS_id: int,
    PAD_id: int,
    tokenizer: ByteLevelBPETokenizer,
    max_len: int = 224,
    min_len: int = 40,
    no_repeat_ngram: int = 3,
    ground_quota: float = 0.40,
    relax_after_quota: bool = True,
    bias_block: float = -6.0,
    whitelist_extra: Optional[Set[int]] = None,
    verbose: bool = False,
) -> str:
    """Greedy decoder that enforces a grounding constraint.

    At each decoding step tokens outside of the source vocabulary are
    penalised until a quota of grounded tokens has been generated.  This
    encourages the model to reuse words from the input.  The decoder also
    blocks early stop and repeated n‑grams for improved fluency.
    Args:
        model: A trained ``HIBERTLike`` instance in eval mode.
        src: Packed source tensor of shape ``(1, S, L)``.
        sm_t: Mask for valid token positions ``(1, S, L)``.
        sm_s: Mask for valid sentences ``(1, S)``.
        BOS_id, EOS_id, PAD_id: Special token IDs.
        tokenizer: The same tokenizer used during training.
        max_len: Maximum number of tokens to generate.
        min_len: Minimum number of tokens to generate before allowing EOS.
        no_repeat_ngram: Block repeated n‑grams of this size.
        ground_quota: Minimum ratio of grounded tokens to total tokens.
        relax_after_quota: If ``True`` the grounding constraint is dropped
            once the quota has been reached.
        bias_block: Logit penalty for tokens outside of the allowed set.
        whitelist_extra: Additional tokens to always allow (e.g. punctuation).
        verbose: Whether to print debug information during decoding.
    Returns:
        A decoded summary (string) with special tokens removed.
    """
    model.eval()
    device = next(model.parameters()).device
    src = src.to(device)
    sm_t = sm_t.to(device) if sm_t is not None else None
    sm_s = sm_s.to(device) if sm_s is not None else None

    V = tokenizer.get_vocab_size()
    special_ids: Set[int] = {int(PAD_id), int(BOS_id), int(EOS_id)}
    src_vocab_ids = _extract_src_vocab_from_tensor(src, sm_t, special_ids)
    # Build initial whitelist: tokens seen in the source or punctuation/common words
    if whitelist_extra is None:
        whitelist_extra = set()
        for tok in [
            ".",
            ",",
            ":",
            ";",
            "-",
            "(",
            ")",
            "/",
            "%",
            "and",
            "or",
            "the",
            "a",
            "an",
            "to",
            "of",
            "in",
            "for",
            "with",
            "on",
            "using",
            "by",
            "from",
            "as",
        ]:
            whitelist_extra.update(_safe_encode(tokenizer, tok))
    allowed_ids_initial = (src_vocab_ids | whitelist_extra) - special_ids
    bias_mask = _build_bias_mask(V, allowed_ids_initial, device, bias_allowed=0.0, bias_block=bias_block)

    seq = [int(BOS_id)]
    grounded_count = 0

    for step in range(max_len):
        dec = torch.tensor(seq, device=device, dtype=torch.long).unsqueeze(0)  # [1,T]
        # compute logits for the last step
        logits = model(src, sm_t, sm_s, dec)[0, -1]  # [V]
        # Prevent EOS until min_len has been reached
        if step < min_len:
            logits[int(EOS_id)] = -1e9
        # Block repeated n‑grams
        if no_repeat_ngram >= 2 and len(seq) >= no_repeat_ngram - 1:
            n = no_repeat_ngram
            cache = {}
            for i in range(len(seq) - n + 1):
                prev = tuple(seq[i : i + n - 1])
                nxt = seq[i + n - 1]
                cache.setdefault(prev, []).append(nxt)
            cur = tuple(seq[-(n - 1) :]) if (n - 1) > 0 else tuple()
            banned = cache.get(cur, [])
            if banned:
                logits[torch.tensor(list(set(banned)), device=device)] = -1e9
        # Apply grounding penalty
        produced = max(1, len(seq) - 1)  # exclude BOS
        ratio_ground = grounded_count / produced
        need_grounding = ratio_ground < ground_quota
        if need_grounding or not relax_after_quota:
            logits = logits + bias_mask
        # Greedy pick
        nxt = int(torch.argmax(logits).item())
        seq.append(nxt)
        if nxt in allowed_ids_initial:
            grounded_count += 1
        if verbose and (step % 20 == 0):
            print(f"[step {step:03d}] ratio_ground={ratio_ground:.2f}, next_id={nxt}")
        if nxt == int(EOS_id):
            break
    # Decode: remove special tokens
    out_ids = [int(i) for i in seq if int(i) not in special_ids]
    try:
        text = tokenizer.decode(out_ids)
    except Exception:
        text = " ".join(map(str, out_ids))
    return text


###############################################################################
#  Public API
###############################################################################

def load_summary_model(model_path: str, tokenizer_dir: str) -> None:
    """Load the tokenizer and summariser model from disk.

    Args:
        model_path: Path to the ``.pth`` checkpoint file containing the model
            weights.  The checkpoint should either be a plain state dict or
            contain a key ``"model"`` with the state dict.
        tokenizer_dir: Directory containing ``vocab.json`` and ``merges.txt``
            produced by the ``ByteLevelBPETokenizer``.  The special tokens
            ``<pad>``, ``<s>`` and ``</s>`` must be present in the vocab.
    Raises:
        FileNotFoundError: if required files are missing.
        RuntimeError: if the tokenizer cannot be loaded.
    """
    global _model, _tokenizer, PAD, BOS, EOS
    # Validate inputs
    model_p = Path(model_path)
    tok_dir = Path(tokenizer_dir)
    if not model_p.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    vocab_path = tok_dir / "vocab.json"
    merges_path = tok_dir / "merges.txt"
    if not vocab_path.is_file() or not merges_path.is_file():
        raise FileNotFoundError(f"Tokenizer files vocab.json/merges.txt missing in {tokenizer_dir}")
    if ByteLevelBPETokenizer is None:
        raise RuntimeError(
            "tokenizers library is not installed.  Please install it via `pip install tokenizers`"
        )
    # Load tokenizer
    tokenizer = ByteLevelBPETokenizer.from_file(str(vocab_path), str(merges_path))
    PAD_id = tokenizer.token_to_id("<pad>")
    BOS_id = tokenizer.token_to_id("<s>")
    EOS_id = tokenizer.token_to_id("</s>")
    if None in (PAD_id, BOS_id, EOS_id):
        raise RuntimeError("Special tokens <pad>, <s> or </s> missing in tokenizer vocab")
    # Optionally set the special tokens map for nicer decoding
    try:
        tokenizer.special_tokens_map = {"pad_token": "<pad>", "bos_token": "<s>", "eos_token": "</s>"}
    except Exception:
        pass
    # Load model state
    state = torch.load(model_p, map_location="cpu")
    # Some checkpoints store the model under a key
    if isinstance(state, dict) and any(isinstance(v, torch.Tensor) for v in state.values()):
        sd = state
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        raise RuntimeError(f"Unexpected checkpoint format for {model_path}")
    vocab_size = tokenizer.get_vocab_size()
    # Instantiate model.  Hyper‑parameters (d_model, layers, etc.) must match those
    # used during training.  If your model differs adjust these values here.
    model = HIBERTLike(vocab_size=vocab_size, pad_id=PAD_id)
    model.load_state_dict(sd)
    model.eval()
    _model = model
    _tokenizer = tokenizer
    PAD, BOS, EOS = PAD_id, BOS_id, EOS_id


def generate_summary(text_en: str) -> str:
    """Generate a summary for a given English document.

    This helper calls ``pack_from_raw_text`` to construct the hierarchical
    representation expected by ``HIBERTLike`` and then uses
    ``generate_grounded`` to produce a summary.  It assumes that
    ``load_summary_model`` has been called previously.

    Args:
        text_en: The English text to summarise.
    Returns:
        A summarised version of the input text.
    Raises:
        RuntimeError: if the model has not been loaded.
    """
    if _model is None or _tokenizer is None or PAD is None or BOS is None or EOS is None:
        raise RuntimeError(
            "Summary model not loaded. Call load_summary_model(model_path, tokenizer_dir) first."
        )
    # Build hierarchical input
    src, sm_t, sm_s = pack_from_raw_text(text_en, _tokenizer, PAD)
    # Run greedy decoding with grounding
    summary = generate_grounded(
        _model,
        src=src,
        sm_t=sm_t,
        sm_s=sm_s,
        BOS_id=BOS,
        EOS_id=EOS,
        PAD_id=PAD,
        tokenizer=_tokenizer,
    )
    return summary