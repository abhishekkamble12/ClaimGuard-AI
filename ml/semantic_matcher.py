"""
ProofPilot — ML Semantic Evidence Matcher
------------------------------------------
Uses TF-IDF Vectorization and Cosine Similarity (scikit-learn) to calculate
continuous document relevance between raw merchant text and evidence descriptions.
Supports both fast pair-wise cosine similarity and corpus-aware pre-fit vocabulary matching.
"""

import functools
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@functools.lru_cache(maxsize=1)
def _get_vectorizer() -> TfidfVectorizer:
    """Cached factory helper for TfidfVectorizer instance."""
    return TfidfVectorizer(stop_words="english")


@functools.lru_cache(maxsize=512)
def compute_semantic_relevance(document_text: str, evidence_description: str) -> float:
    """
    Calculate Cosine Similarity between document text and requirement description.
    Uses LRU caching for high performance across evaluation benchmarks.
    Returns float score between 0.0 and 1.0.
    """
    if not document_text or not str(document_text).strip():
        return 0.0

    doc_clean = str(document_text).strip()
    desc_clean = str(evidence_description).strip()

    try:
        vectorizer = _get_vectorizer()
        tfidf_matrix = vectorizer.fit_transform([doc_clean, desc_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity), 4)
    except Exception:
        # Fallback if text is too short or contains only stop words
        return 0.5 if len(doc_clean) > 10 else 0.0


class CorpusAwareSemanticMatcher:
    """
    Corpus-level semantic relevance engine. Fits a vocabulary across all payment
    reason-code evidence descriptions in the platform config for consistent,
    comparable TF-IDF embeddings.
    """

    def __init__(self, reason_code_configs: dict[str, Any] | None = None):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True,
        )
        self.is_fit = False
        if reason_code_configs:
            self.fit_corpus(reason_code_configs)

    def fit_corpus(self, reason_code_configs: dict[str, Any]) -> None:
        corpus = []
        for rc_id, conf in reason_code_configs.items():
            if isinstance(conf, dict):
                req = conf.get("required_evidence", {})
                for eid, edetail in req.items():
                    if isinstance(edetail, dict):
                        desc = edetail.get("description", "")
                        examples = " ".join(edetail.get("examples", []))
                        corpus.append(f"{eid.replace('_', ' ')} {desc} {examples}")
                    else:
                        corpus.append(eid.replace("_", " "))

        if corpus:
            try:
                self.vectorizer.fit(corpus)
                self.is_fit = True
            except Exception:
                self.is_fit = False

    def compute_relevance(self, document_text: str, evidence_description: str) -> float:
        """Compute cosine similarity using the corpus-aware pre-fit vocabulary."""
        if not document_text or not str(document_text).strip():
            return 0.0

        if not self.is_fit:
            return compute_semantic_relevance(document_text, evidence_description)

        try:
            vecs = self.vectorizer.transform([str(document_text).strip(), str(evidence_description).strip()])
            sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
            return round(float(sim), 4)
        except Exception:
            return compute_semantic_relevance(document_text, evidence_description)
