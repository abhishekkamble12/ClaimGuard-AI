"""
ProofPilot — ML Semantic Evidence Matcher
------------------------------------------
Uses TF-IDF Vectorization and Cosine Similarity (scikit-learn) to calculate
continuous document relevance between raw merchant text and evidence descriptions.
Supports both fast pair-wise cosine similarity and corpus-aware pre-fit vocabulary matching.
"""

import functools
import math
import os
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_st_model = None
_HAS_SENTENCE_TRANSFORMERS = False

try:
    from sentence_transformers import SentenceTransformer
    # Check if disabled by env var (e.g. for lightweight test runs)
    if os.getenv("USE_TFIDF_ONLY", "0") != "1":
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    _st_model = None
    _HAS_SENTENCE_TRANSFORMERS = False


def _create_vectorizer() -> TfidfVectorizer:
    """Factory helper to create a fresh TfidfVectorizer instance."""
    return TfidfVectorizer(stop_words="english")


@functools.lru_cache(maxsize=512)
def compute_semantic_relevance(document_text: str, evidence_description: str) -> float:
    """
    Calculate semantic similarity between document text and requirement description.
    Uses dense SentenceTransformer embeddings (all-MiniLM-L6-v2) when available,
    falling back to TF-IDF cosine similarity for low-latency / zero-dependency setups.
    Returns float score between 0.0 and 1.0.
    """
    if not document_text or not document_text.strip():
        return 0.0
    if not evidence_description or not evidence_description.strip():
        return 0.0

    doc_clean = document_text.strip()
    desc_clean = evidence_description.strip()

    # 1. Primary path: Dense sentence embeddings (captures semantics, synonyms, paraphrasing)
    if _HAS_SENTENCE_TRANSFORMERS and _st_model is not None:
        try:
            embeddings = _st_model.encode([doc_clean, desc_clean])
            sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            sim_val = float(sim)
            if not (math.isnan(sim_val) or math.isinf(sim_val)):
                return round(max(0.0, min(1.0, sim_val)), 4)
        except Exception:
            pass

    # 2. Fallback path: TF-IDF cosine similarity
    try:
        vectorizer = _create_vectorizer()
        tfidf_matrix = vectorizer.fit_transform([doc_clean, desc_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        sim_val = float(similarity)
        if math.isnan(sim_val) or math.isinf(sim_val):
            return 0.0
        return round(max(0.0, min(1.0, sim_val)), 4)
    except Exception:
        # Fallback without stop words if document consists of short tokens or domain numbers
        try:
            vectorizer_fallback = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tfidf_matrix = vectorizer_fallback.fit_transform([doc_clean, desc_clean])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            sim_val = float(similarity)
            if math.isnan(sim_val) or math.isinf(sim_val):
                return 0.0
            return round(max(0.0, min(1.0, sim_val)), 4)
        except Exception:
            return 0.0


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
        if not reason_code_configs or not isinstance(reason_code_configs, dict):
            return

        corpus: list[str] = []
        for _, conf in reason_code_configs.items():
            if isinstance(conf, dict):
                req = conf.get("required_evidence", {})
                if isinstance(req, dict):
                    for eid, edetail in req.items():
                        eid_str = str(eid).replace("_", " ")
                        if isinstance(edetail, dict):
                            desc = edetail.get("description", "")
                            raw_examples = edetail.get("examples", [])
                            examples = " ".join(raw_examples) if isinstance(raw_examples, list) else str(raw_examples or "")
                            text = f"{eid_str} {desc} {examples}".strip()
                            if text:
                                corpus.append(text)
                        elif isinstance(edetail, str):
                            text = f"{eid_str} {edetail}".strip()
                            if text:
                                corpus.append(text)
                        else:
                            corpus.append(eid_str)
                elif isinstance(req, list):
                    for eid in req:
                        corpus.append(str(eid).replace("_", " "))

        if corpus:
            try:
                self.vectorizer.fit(corpus)
                self.is_fit = True
            except Exception:
                self.is_fit = False

    def compute_relevance(self, document_text: str, evidence_description: str) -> float:
        """Compute cosine similarity using the corpus-aware pre-fit vocabulary."""
        if not document_text or not document_text.strip():
            return 0.0
        if not evidence_description or not evidence_description.strip():
            return 0.0

        doc_clean = document_text.strip()
        desc_clean = evidence_description.strip()

        if not self.is_fit:
            return compute_semantic_relevance(doc_clean, desc_clean)

        try:
            vecs = self.vectorizer.transform([doc_clean, desc_clean])
            sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
            sim_val = float(sim)
            if math.isnan(sim_val) or math.isinf(sim_val):
                sim_val = 0.0
            if sim_val == 0.0:
                # If domain words in the document/description were not present in the reason code corpus,
                # fall back to pairwise similarity calculation.
                return compute_semantic_relevance(doc_clean, desc_clean)
            return round(max(0.0, min(1.0, sim_val)), 4)
        except Exception:
            return compute_semantic_relevance(doc_clean, desc_clean)
