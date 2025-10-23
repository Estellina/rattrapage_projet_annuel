# app/models/pretrained_summary_mbart.py
from __future__ import annotations
from pathlib import Path
import torch

from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

try:
    from safetensors.torch import load_file as load_safetensors
except Exception:
    load_safetensors = None  # fallback to .bin loading

# Liste des tokens de contrôle (préfixes) utilisés pour le feedback
CONTROL_TOKENS = [
    "<LEN_SHORT>", "<LEN_MEDIUM>", "<LEN_LONG>",
    "<TONE_FORMAL>", "<TONE_NEUTRAL>", "<TONE_CASUAL>",
    "<FOCUS_GENERAL>", "<FOCUS_DETAILS>", "<FOCUS_RESULTS>",
    "<FOCUS_METHODS>", "<FOCUS_LIMITATIONS>", "<FOCUS_APPLICATIONS>",
    "<STRUCT_PARAGRAPHS>", "<STRUCT_BULLETS>", "<STRUCT_SECTIONS>",
    "<COVER_KEYPOINTS>", "<COVER_COMPREHENSIVE>",
    "<STYLE_ABSTRACTIVE>", "<STYLE_EXTRACTIVE>",
    "<NUM_KEEP>", "<NUM_MINIMIZE>",
    "<CITE_INCLUDE>", "<CITE_EXCLUDE>",
    "<DOC_START>",
]

_model: AutoModelForSeq2SeqLM | None = None
_tokenizer: AutoTokenizer | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _has(path: Path, *names: str) -> bool:
    return any((path / n).exists() for n in names)

def _load_state_dict(weights_dir: Path) -> dict:
    safep = weights_dir / "model.safetensors"
    binp  = weights_dir / "pytorch_model.bin"
    if safep.exists() and load_safetensors:
        return load_safetensors(str(safep), device="cpu")
    if binp.exists():
        return torch.load(str(binp), map_location="cpu")
    raise FileNotFoundError("No weights found (model.safetensors or pytorch_model.bin)")

def _fix_generation_config(cfg):
    if getattr(cfg, "early_stopping", None) is None:
        cfg.early_stopping = True
    return cfg

def _ensure_control_tokens(tokenizer, model, tokens=CONTROL_TOKENS):
    vocab = tokenizer.get_vocab()
    missing = [t for t in tokens if t not in vocab]
    if not missing:
        return False
    add_spec = {"additional_special_tokens": list(dict.fromkeys(
        (tokenizer.special_tokens_map.get("additional_special_tokens") or []) + tokens
    ))}
    tokenizer.add_special_tokens(add_spec)
    model.resize_token_embeddings(len(tokenizer))
    return True

def setup_with_local_or_hf(
    local_dir: str | Path,
    hf_id: str = "facebook/mbart-large-50",
    use_fast_tokenizer: bool = False,
    tgt_lang: str = "fr_XX",
):
    """
    Charge un checkpoint local ou fallback HF, en ajoutant les tokens de feedback et en corrigeant
    la génération si nécessaire.
    - local_dir: dossier contenant config.json, model.safetensors/pytorch_model.bin, tokenizer.json, etc.
    - hf_id: modèle de fallback (par ex. facebook/mbart-large-50)
    - use_fast_tokenizer: False permet d’éviter la dépendance à protobuf
    - tgt_lang: langue de sortie forcée (fr_XX)
    """
    global _model, _tokenizer

    local_dir = Path(local_dir)
    # Choix du tokenizer : local si disponible, sinon HF
    tok_src = str(local_dir) if _has(local_dir, "tokenizer.json", "spiece.model", "vocab.json") else hf_id
    # 1) charge le tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=use_fast_tokenizer)

    # 2) tentative locale
    try:
        cfg = AutoConfig.from_pretrained(str(local_dir))
        cfg = _fix_generation_config(cfg)
        mdl = AutoModelForSeq2SeqLM.from_config(cfg)
        state = _load_state_dict(local_dir)
        mdl.load_state_dict(state, strict=False)
        _model = mdl
    except Exception:
        # 3) fallback HF complet
        _tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=use_fast_tokenizer)
        _model = AutoModelForSeq2SeqLM.from_pretrained(hf_id)

    # 4) assure la présence des tokens de feedback
    _ensure_control_tokens(_tokenizer, _model, CONTROL_TOKENS)

    # 5) langue cible pour mBART
    if hasattr(_tokenizer, "lang_code_to_id"):
        _model.config.forced_bos_token_id = _tokenizer.lang_code_to_id.get(tgt_lang, _model.config.forced_bos_token_id)

    # 6) device + cache
    _model.to(_device).eval()
    _model.config.use_cache = True
    return _model, _tokenizer, _device

def get_model_and_tokenizer():
    return _model, _tokenizer, _device
