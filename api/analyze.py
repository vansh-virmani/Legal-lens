from fastapi import APIRouter
from pydantic import BaseModel
import os
import pickle
import faiss

# ML modules
from ml.embeddings import embed_query
from ml.retrieval import retrieve_top_k
from ml.rag import build_prompt, _load_generator
from ml.risk import analyze_risks
from ml.summary import summarize
from ml.timeline import build_timeline

router = APIRouter()

INDEX_DIR = "data/indexes"


# -------- REQUEST MODEL --------
class QueryRequest(BaseModel):
    doc_id: str
    query: str


# -------- ANALYZE ENDPOINT --------
@router.post("/analyze")
def analyze(req: QueryRequest):

    path_index = f"{INDEX_DIR}/{req.doc_id}.faiss"
    path_data = f"{INDEX_DIR}/{req.doc_id}.pkl"

    if not os.path.exists(path_index):
        return {"error": "Document not found"}

    # -------- LOAD --------
    index = faiss.read_index(path_index)

    with open(path_data, "rb") as f:
        data = pickle.load(f)

    chunks = data["chunks"]
    text = data["text"]

    query = req.query.lower()

    # -------- ROUTING --------

    # 🔹 RISK
    if "risk" in query:
        result = analyze_risks(chunks)
        return {"type": "risk", "result": result}

    # 🔹 SUMMARY
    elif "summary" in query:
        result = summarize(text)
        return {"type": "summary", "result": result}

    # 🔹 TIMELINE
    elif "timeline" in query or "date" in query:
        result = build_timeline(text)
        return {"type": "timeline", "result": result}

    # 🔹 DEFAULT → RAG Q&A
    else:
        q_emb = embed_query(req.query)

        D, I = index.search(q_emb.reshape(1, -1), 3)
        retrieved = [chunks[i] for i in I[0]]

        context = "\n".join(retrieved)

        model, tokenizer = _load_generator()
        prompt = build_prompt(context, req.query)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=150)

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "type": "rag",
            "answer": answer,
            "context": retrieved
        }