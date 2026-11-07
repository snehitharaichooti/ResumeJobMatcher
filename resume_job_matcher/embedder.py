"""Text embeddings: Sentence-BERT when available, otherwise a TF-IDF fallback."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder:
    """backend: 'auto' (SBERT -> TF-IDF fallback), 'sbert' (error if unavailable), 'tfidf'."""

    def __init__(self, backend: str = "auto", model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.name = "TF-IDF (scikit-learn)"
        self.load_error: str | None = None
        self.vectorizer: TfidfVectorizer | None = None
        if backend in ("auto", "sbert"):
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self.name = f"Sentence-BERT ({model_name})"
            except Exception as exc:  # missing package, no internet, etc.
                self.load_error = f"{type(exc).__name__}: {exc}"
                if backend == "sbert":
                    raise

    @property
    def is_neural(self) -> bool:
        return self.model is not None

    @property
    def calibration(self) -> tuple[float, float]:
        """Raw cosine range mapped to 0..1 (SBERT cosines for related text are ~0.3-0.7)."""
        return (0.15, 0.75) if self.is_neural else (0.0, 0.45)

    def prepare(self, corpus: list[str]) -> None:
        if not self.is_neural:
            self.vectorizer = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b[\w+#./]+\b")
            self.vectorizer.fit(corpus)

    def encode(self, texts: list[str]):
        if self.is_neural:
            return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return self.vectorizer.transform(texts)
