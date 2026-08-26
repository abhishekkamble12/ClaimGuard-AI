"""
ProofPilot — ML Semantic Evidence Matcher
------------------------------------------
Uses TF-IDF Vectorization and Cosine Similarity (scikit-learn) to calculate
continuous document relevance between raw merchant text and evidence descriptions.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_semantic_relevance(document_text: str, evidence_description: str) -> float:
    """
    Calculate Cosine Similarity between document text and requirement description.
    Returns float score between 0.0 and 1.0.
    """
    if not document_text or not str(document_text).strip():
        return 0.0

    doc_clean = str(document_text).strip()
    desc_clean = str(evidence_description).strip()

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([doc_clean, desc_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity), 4)
    except Exception:
        # Fallback if text is too short or contains only stop words
        return 0.5 if len(doc_clean) > 10 else 0.0
