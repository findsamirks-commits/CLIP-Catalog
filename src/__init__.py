"""CLIP-Catalog: Vision-Language Zero-Shot Architecture for Retail Catalogs.

Exposes contrastive representations, FAISS retrieval index, and commercial
onboarding bridge for FMCG catalog intelligence.
"""

__version__ = "0.1.0"
__author__ = "Samir Sandhwar"

from src.onboarding_bridge import OnboardingBridge
from src.eval import compute_retrieval_metrics, compute_zero_shot_accuracy

__all__ = [
    "OnboardingBridge",
    "compute_retrieval_metrics",
    "compute_zero_shot_accuracy",
]

