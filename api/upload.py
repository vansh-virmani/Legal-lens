from fastapi import APIRouter, UploadFile, File
import uuid
import os
import pickle
import faiss
import fitz  # PyMuPDF


# your ML modules
from ml.chunking import chunk_by_clause
from ml.embeddings import embed_text

router = APIRouter()

INDEX_DIR = "data/indexes"
os.makedirs(INDEX_DIR, exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())

    # -------- READ PDF --------
    pdf = fitz.open(stream=await file.read(), filetype="pdf")
    text = "".join([page.get_text() for page in pdf])

    # -------- CHUNK --------
    chunks = chunk_by_clause(text)

    if not chunks:
        return {"error": "No text extracted"}

    # -------- EMBEDDING --------
    embeddings = embed_text(chunks)

    # -------- FAISS INDEX --------
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # -------- SAVE --------
    faiss.write_index(index, f"{INDEX_DIR}/{doc_id}.faiss")

    with open(f"{INDEX_DIR}/{doc_id}.pkl", "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "text": text
        }, f)

    return {
        "message": "uploaded & indexed",
        "doc_id": doc_id,
        "total_chunks": len(chunks)
    }