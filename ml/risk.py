from transformers import pipeline

_classifier = None


# ── Labels ────────────────────────────────────────────────────────────────
RISK_LABELS = [
    "high risk liability clause",
    "high risk indemnification clause",
    "medium risk termination clause",
    "medium risk penalty clause",
    "low risk standard clause",
]

_LABEL_TO_LEVEL = {
    "high risk liability clause": ("High", "Liability"),
    "high risk indemnification clause": ("High", "Indemnification"),
    "medium risk termination clause": ("Medium", "Termination"),
    "medium risk penalty clause": ("Medium", "Penalty"),
    "low risk standard clause": ("Low", "Standard"),
}


# ── Lightweight Classifier Loader ─────────────────────────────────────────
def _load_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3",  # ~400MB instead of 1.6GB
        )
    return _classifier


# ── ML Risk Detection ─────────────────────────────────────────────────────
def detect_risk_ml(clause: str):
    try:
        clf = _load_classifier()
        out = clf(clause, RISK_LABELS, multi_label=False)

        top_label = out["labels"][0]
        confidence = out["scores"][0]

        level, rtype = _LABEL_TO_LEVEL.get(top_label, ("Low", "Unknown"))

        return level, rtype, round(float(confidence), 3)

    except Exception:
        return detect_risk_keyword(clause)


# ── Keyword Fallback ──────────────────────────────────────────────────────
def detect_risk_keyword(clause: str):
    c = clause.lower()

    if any(w in c for w in ("not liable", "no liability", "indemnif", "consequential")):
        return "High", "Liability", 1.0

    if any(w in c for w in ("terminate", "termination")):
        return "Medium", "Termination", 1.0

    if any(w in c for w in ("penalty", "breach", "damages")):
        return "High", "Penalty", 1.0

    return "Low", "Standard", 1.0


# ── Public API ────────────────────────────────────────────────────────────
def detect_risk(clause: str):
    level, _, _ = detect_risk_keyword(clause)
    return level


def analyze_risks(chunks: list[str], use_ml: bool = True) -> list[dict]:
    results = []

    for chunk in chunks:
        if use_ml:
            level, rtype, conf = detect_risk_ml(chunk)
            method = "ml"
        else:
            level, rtype, conf = detect_risk_keyword(chunk)
            method = "keyword"

        results.append({
            "clause": chunk,
            "risk": level,
            "risk_type": rtype,
            "confidence": conf,
            "method": method,
        })

    return results