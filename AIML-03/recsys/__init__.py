"""Two-stage recommender: FAISS candidate retrieval + TFRS ranking."""
import os

# TensorFlow Recommenders needs Keras 2 (provided by the tf-keras package).
# This must be set before TensorFlow is imported anywhere.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
