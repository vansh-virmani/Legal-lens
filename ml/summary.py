from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_summarizer = None
_tokenizer = None


def _load_summarizer():
    global _summarizer, _tokenizer
    if _summarizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        _summarizer = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return _summarizer, _tokenizer


def _generate(prompt: str, max_tokens=150) -> str:
    model, tokenizer = _load_summarizer()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ── PUBLIC API ─────────────────────────────────────────────

def summarize(text: str, mode: str = "brief"):
    
    # 🔹 Brief → return string
    if mode == "brief":
        prompt = f"Summarize this legal document briefly:\n\n{text}"
        return _generate(prompt)

    # 🔹 Detailed → return dict (FIXED)
    elif mode == "detailed":
        aspects = {
            "Parties": "Who are the parties involved?",
            "Financial": "What are the financial obligations?",
            "Termination": "What are the termination conditions?",
            "Liability": "What liabilities exist?",
        }

        result = {}

        for key, question in aspects.items():
            prompt = f"{question}\n\n{text}"
            result[key] = _generate(prompt, max_tokens=120)

        return result

    # fallback
    else:
        return _generate(f"Summarize:\n{text}")