import re

# ── Date / duration patterns ──────────────────────────────────────────────────

_DATE_PATTERNS: list[tuple[str, str]] = [
    (r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",                        "numeric date"),
    (r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
     r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
     r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",          "full date"),
    (r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|"
     r"August|September|October|November|December)\s+\d{4}\b",        "full date"),
    (r"\bEffective\s+Date\b",                                          "effective date"),
    (r"\b\d+\s+(?:calendar\s+)?(?:days?|months?|years?)\b",           "duration"),
    (r"\bone\s+year\b|\btwo\s+years?\b|\bthree\s+years?\b",           "duration"),
]

_OBLIGATION_KEYWORDS = [
    "shall", "must", "will", "agree", "agreed", "required", "obligated",
    "responsible", "liable", "entitled", "permitted", "may not", "cannot",
]


# ── Extraction helpers ────────────────────────────────────────────────────────

def extract_dates(text: str) -> list[dict]:
    """Find all date/duration references and their character positions."""
    found = []
    seen_spans: set[tuple[int, int]] = set()

    for pattern, label in _DATE_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            span = (m.start(), m.end())
            if span not in seen_spans:
                seen_spans.add(span)
                found.append({
                    "reference": m.group().strip(),
                    "type":      label,
                    "position":  m.start(),
                })

    found.sort(key=lambda x: x["position"])
    return found


def extract_obligations(text: str) -> list[dict]:
    """
    Return sentences that contain obligation/permission language.
    Each entry includes the matched keyword for explainability.
    """
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    obligations = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        lower = sent.lower()
        for kw in _OBLIGATION_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", lower):
                obligations.append({
                    "text":               sent,
                    "obligation_keyword": kw,
                })
                break  # one match per sentence is enough

    return obligations


# ── Public API ────────────────────────────────────────────────────────────────

def build_timeline(document_text: str) -> dict:
    """
    Build a structured timeline from a legal document.

    Returns
    -------
    dict
        dates        : list[dict]  — each date/duration reference found
        obligations  : list[dict]  — sentences containing obligation verbs
        summary      : dict        — high-level counts
    """
    dates       = extract_dates(document_text)
    obligations = extract_obligations(document_text)

    return {
        "dates":       dates,
        "obligations": obligations,
        "summary": {
            "total_date_references":  len(dates),
            "total_obligations":      len(obligations),
            "duration_references":    [d for d in dates if d["type"] == "duration"],
        },
    }