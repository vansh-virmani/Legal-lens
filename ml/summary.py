from transformers import pipeline

_summarizer = None

ASPECTS = [
    ("parties",      "Who are the parties involved in this agreement?"),
    ("duration",     "What is the duration or term of this agreement?"),
    ("termination",  "How and when can this agreement be terminated?"),
    ("liability",    "What are the liability limitations or exclusions?"),
    ("disputes",     "How are disputes or conflicts resolved?"),
]


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_summarizer = None
_tokenizer = None


def _load_summarizer():
    global _summarizer, _tokenizer
    if _summarizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        _summarizer = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return _summarizer, _tokenizer


def summarize(text: str, mode: str = "brief") -> str:
    model, tokenizer = _load_summarizer()

    if mode == "brief":
        prompt = f"Summarize this legal document briefly:\n\n{text}"
    elif mode == "detailed":
        prompt = f"Provide a detailed structured summary of this legal document:\n\n{text}"
    else:
        prompt = f"Summarize:\n{text}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def summarize_clause(clause_text: str) -> str:
    """Explain a single clause in plain English."""
    model = _load_summarizer()
    prompt = f"Explain this legal clause in simple language:\n{clause_text}\n\nSimple explanation:"
    out = model(prompt, max_new_tokens=100, do_sample=False)
    return out[0]["generated_text"].strip()