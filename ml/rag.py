from ml.chunking import chunk_by_clause, chunk_text
from ml.embeddings import embed_text, embed_query
from ml.retrieval import retrieve_top_k

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_generator = None
_tokenizer = None


# ── Generator Loader (FLAN-T5 correct usage) ────────────────────────────────
def _load_generator():
    global _generator, _tokenizer
    if _generator is None:
        _tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        _generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return _generator, _tokenizer


# ── Prompt Builder ─────────────────────────────────────────────────────────
def build_prompt(context: str, query: str) -> str:
    return (
        "You are a legal assistant.\n"
        "Answer ONLY using the context below.\n"
        "Be concise and precise.\n"
        'If the answer is not present, say "Not found in document."\n\n'
        f"{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


# ── Main RAG Pipeline ──────────────────────────────────────────────────────
def get_answer(
    document_text: str,
    query: str,
    k: int = 3,
    use_clause_chunking: bool = True,
) -> dict:

    # 1. Chunk
    chunks = (
        chunk_by_clause(document_text)
        if use_clause_chunking
        else chunk_text(document_text)
    )

    if not chunks:
        return {
            "answer": "No content found.",
            "retrieved_chunks": [],
            "scores": [],
            "citations": [],
            "total_chunks": 0,
        }

    # 2. Embed
    chunk_embeddings = embed_text(chunks)
    query_embedding = embed_query(query)

    # 3. Retrieve
    k = min(k, len(chunks))
    top_chunks, scores = retrieve_top_k(
        query_embedding, chunk_embeddings, chunks, k=k
    )

    # 4. Build context + citations
    context_lines = [f"[{i+1}] {chunk}" for i, chunk in enumerate(top_chunks)]
    context = "\n".join(context_lines)

    citations = [
        {"id": i + 1, "text": chunk, "score": float(scores[i])}
        for i, chunk in enumerate(top_chunks)
    ]

    # 5. Generate (correct FLAN usage)
    prompt = build_prompt(context, query)
    model, tokenizer = _load_generator()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return {
        "answer": answer,
        "retrieved_chunks": top_chunks,
        "scores": scores,
        "citations": citations,
        "total_chunks": len(chunks),
    }