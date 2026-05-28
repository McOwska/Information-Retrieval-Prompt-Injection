from __future__ import annotations

import joblib
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

MODEL_REPO = "belrem/llm-prompt-intent-classifier"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ADVERSARIAL_LABEL_ID = 3

# Loaded lazily on first call to is_adversarial(), or eagerly via load_classifier().
_embedder: SentenceTransformer | None = None
_clf = None


def load_classifier() -> tuple[SentenceTransformer, object]:
    """
    Downloads and loads the embedding model and the scikit-learn classifier.
    Safe to call multiple times — returns the cached instances after the first load.
    """
    global _embedder, _clf

    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)

    if _clf is None:
        clf_path = hf_hub_download(repo_id=MODEL_REPO, filename="classifier.joblib")
        _clf = joblib.load(clf_path)

    return _embedder, _clf


def is_adversarial(text: str) -> bool:
    """
    Returns True if the text is classified as adversarial intent.

    Designed to be passed directly as the classifier argument to
    run_multihop_agent() or apply_classifier_guard().

    The underlying model is belrem/llm-prompt-intent-classifier, a
    LogisticRegression head on all-MiniLM-L6-v2 sentence embeddings,
    with four classes: creative (0), informational (1), task (2), adversarial (3).

    Args:
        text: document text to screen

    Returns:
        True if the model predicts adversarial intent, False otherwise
    """
    embedder, clf = load_classifier()
    vec = embedder.encode([text], convert_to_numpy=True)
    label_id = int(clf.predict(vec)[0])
    return label_id == ADVERSARIAL_LABEL_ID
