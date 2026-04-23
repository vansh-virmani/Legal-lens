import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model once
_model = SentenceTransformer("all-MiniLM-L6-v2")
_cache: dict[str, np.ndarray] = {}


def _key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def embed_text(text_list: list[str], use_cache: bool = True) -> np.ndarray:
    """
    Encode a list of strings → 2-D float32 array (N × 384).
    Hits cache for previously seen strings.
    """
    if not use_cache:
        return _model.encode(text_list, show_progress_bar=False)

    to_embed, seen = [], {}
    for t in text_list:
        k = _key(t)
        if k in _cache:
            seen[t] = _cache[k]
        else:
            to_embed.append(t)

    if to_embed:
        new_embs = _model.encode(to_embed, show_progress_bar=False)
        for t, e in zip(to_embed, new_embs):
            _cache[_key(t)] = e
            seen[t] = e

    return np.array([seen[t] for t in text_list])


def embed_query(query: str) -> np.ndarray:
    """Encode a single query string → 1-D float32 vector (384,)."""
    k = _key(query)
    if k not in _cache:
        _cache[k] = _model.encode([query])[0]
    return _cache[k]