import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize


def chunk_text(text, window_size=2, overlap=1):
    """
    Sliding-window chunker over NLTK sentences.

    Args:
        text        : raw document string
        window_size : sentences per chunk  (default 2)
        overlap     : sentences shared between adjacent chunks (default 1)

    Returns:
        list[str]  — overlapping multi-sentence chunks
    """
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= window_size:
        return sentences

    chunks = []
    step = max(1, window_size - overlap)

    for i in range(0, len(sentences) - window_size + 1, step):
        window = sentences[i : i + window_size]
        chunks.append(" ".join(window))

    return chunks


def chunk_by_clause(text):
    """
    Paragraph-aware clause chunker for legal documents.

    Splits on blank lines first (preserving clause groupings),
    then sentence-tokenises within each paragraph.

    Returns:
        list[str]  — one entry per legal sentence/clause
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        sentences = sent_tokenize(para)
        chunks.extend(s.strip() for s in sentences if s.strip())
    return chunks