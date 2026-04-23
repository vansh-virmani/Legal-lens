import json
from ml.rag import get_answer
from ml.risk  import analyze_risks
from ml.summary         import summarize
from ml.timeline        import build_timeline

# ─────────────────────────────────────────────────────────────────────────────
# Load sample document
# ─────────────────────────────────────────────────────────────────────────────
with open("data/sample.txt","r",encoding="utf-8") as f:
    text = f.read()

print("=" * 60)
print("LEGAL_LENS V2")
print("=" * 60)
print(f"\nDocument ({len(text)} chars):\n{text}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. RAG — Q&A with citations
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("1. RAG — QUESTION ANSWERING WITH CITATIONS")
print("─" * 60)

query = "How can the agreement be terminated?"
result = get_answer(text, query, k=3)

print(f"\nQuery   : {query}")
print(f"Answer  : {result['answer']}")
print(f"\nCitations (ranked by relevance):")
for c in result["citations"]:
    print(f"  [{c['id']}] score={c['score']:.4f}  \"{c['text']}\"")
print(f"\nTotal chunks in document: {result['total_chunks']}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Risk Analysis — ML-based, with confidence + type
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("2. RISK ANALYSIS  (zero-shot classification)")
print("─" * 60)

risks = analyze_risks(result["retrieved_chunks"], use_ml=True)
for r in risks:
    print(f"\n  Clause     : {r['clause']}")
    print(f"  Risk Level : {r['risk']}  |  Type: {r['risk_type']}  |  Confidence: {r['confidence']}  |  Method: {r['method']}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Summarisation — brief + detailed aspects
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("3. SUMMARISATION  (brief + multi-aspect)")
print("─" * 60)

brief = summarize(text, mode="brief")
print(f"\nBrief summary:\n  {brief}")

detailed = summarize(text, mode="detailed")
print("\nDetailed aspect-by-aspect summary:")
for aspect, answer in detailed.items():
    print(f"  {aspect.upper():12s}: {answer}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Timeline Extraction
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4. TIMELINE EXTRACTION")
print("─" * 60)

timeline = build_timeline(text)

print(f"\nDate / duration references found: {timeline['summary']['total_date_references']}")
for d in timeline["dates"]:
    print(f"  [{d['type']}]  \"{d['reference']}\"  (pos {d['position']})")

print(f"\nObligation statements found: {timeline['summary']['total_obligations']}")
for o in timeline["obligations"]:
    print(f"  keyword='{o['obligation_keyword']}'  →  \"{o['text']}\"")

print("\n" + "=" * 60)
print("V2 PIPELINE COMPLETE")
print("=" * 60)