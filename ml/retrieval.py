import numpy as np

try:
    import faiss
    _FAISS = True
except ImportError:
    _FAISS = False


# ── FAISS path ────────────────────────────────────────────────────────────────

def _faiss_search(query_emb, chunk_embs, k):
    embs = np.array(chunk_embs, dtype="float32")
    faiss.normalize_L2(embs)

    index = faiss.IndexFlatIP(embs.shape[1])  # cosine via normalised IP
    index.add(embs)

    q = np.array([query_emb], dtype="float32")
    faiss.normalize_L2(q)

    scores, idxs = index.search(q, k)
    return idxs[0].tolist(), scores[0].tolist()


# ── NumPy fallback ─────────────────────────────────────────────────────────────

def _numpy_search(query_emb, chunk_embs, k):
    embs = np.array(chunk_embs)
    q = np.array(query_emb)

    norms = np.linalg.norm(embs, axis=1)
    sims = embs.dot(q) / (norms * np.linalg.norm(q) + 1e-10)

    top_idx = np.argsort(sims)[-k:][::-1].tolist()
    top_scores = [float(sims[i]) for i in top_idx]
    return top_idx, top_scores


# ── Public API ─────────────────────────────────────────────────────────────────

def retrieve_top_k(query_embedding, chunk_embeddings, chunks, k=3):
    """
    Return the top-k most relevant chunks and their cosine scores.

    Args:
        query_embedding  : 1-D array (dim,)
        chunk_embeddings : 2-D array (N, dim)
        chunks           : list[str] of length N
        k                : number of results

    Returns:
        top_chunks : list[str]
        top_scores : list[float]   cosine similarity in [−1, 1]
    """
    k = min(k, len(chunks))
    search = _faiss_search if _FAISS else _numpy_search
    idxs, scores = search(query_embedding, chunk_embeddings, k)

    top_chunks = [chunks[i] for i in idxs]
    return top_chunks, [round(s, 4) for s in scores]