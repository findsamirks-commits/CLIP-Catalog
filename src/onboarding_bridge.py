"""Onboarding Bridge: Commercial business decision layer for autonomous SKU onboarding.

Decouples zero-shot raw model probabilities from retail workflow actions:
- auto_tag: Direct automatic catalog publication (zero human labor)
- human_review: 3-second single-click verification queue
- reject: Hard stop (forbidden categories or below minimum confidence)
- escalate: Ambiguous low-margin or policy-conflict cases routed to Category Head
"""

from typing import Any, Dict, List, Optional, Tuple


class OnboardingBridge:
    """Business policy layer translating raw zero-shot logits into enterprise catalog actions."""

    def __init__(
        self,
        auto_tag_confidence: float = 0.70,
        review_queue_confidence: float = 0.40,
        min_top1_margin: float = 0.15,
        forbidden_categories: Optional[List[str]] = None
    ):
        self.auto_tag_confidence = float(auto_tag_confidence)
        self.review_queue_confidence = float(review_queue_confidence)
        self.min_top1_margin = float(min_top1_margin)
        self.forbidden_categories = [c.lower() for c in (forbidden_categories or [])]

    def decide(
        self,
        zero_shot_scores: Dict[str, float],
        sku_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluates model prediction scores against business guardrails.

        Args:
            zero_shot_scores: Dict mapping category name to float probability [0.0, 1.0].
            sku_metadata: Optional dictionary with SKU details (e.g., title, brand, supplier).

        Returns:
            Dict containing:
                action: 'auto_tag' | 'human_review' | 'reject' | 'escalate'
                category: Top-predicted category string
                confidence: Top-1 float probability
                margin: Difference between top-1 and top-2 probability
                audit: List of explanatory decision audit strings
        """
        sku_metadata = sku_metadata or {}
        audit_log: List[str] = []

        if not zero_shot_scores:
            return {
                "action": "reject",
                "category": "Unknown",
                "confidence": 0.0,
                "margin": 0.0,
                "audit": ["Zero prediction scores provided."]
            }

        # Sort scores descending
        sorted_scores = sorted(zero_shot_scores.items(), key=lambda kv: kv[1], reverse=True)
        top1_cat, top1_score = sorted_scores[0]
        top2_cat, top2_score = sorted_scores[1] if len(sorted_scores) > 1 else ("None", 0.0)
        margin = float(top1_score - top2_score)

        # 1. Check forbidden categories (hard compliance block)
        if top1_cat.lower() in self.forbidden_categories:
            audit_log.append(f"Hard block: Category '{top1_cat}' is in forbidden categories.")
            return {
                "action": "reject",
                "category": top1_cat,
                "confidence": float(top1_score),
                "margin": margin,
                "audit": audit_log
            }

        # 2. Check title or metadata for forbidden compliance triggers
        title = str(sku_metadata.get("title", "")).lower()
        for forbidden in self.forbidden_categories:
            if forbidden in title:
                audit_log.append(f"Hard block: SKU title contains forbidden token '{forbidden}'.")
                return {
                    "action": "reject",
                    "category": top1_cat,
                    "confidence": float(top1_score),
                    "margin": margin,
                    "audit": audit_log
                }

        # 3. Floor threshold check
        if top1_score < self.review_queue_confidence:
            audit_log.append(
                f"Confidence floor violation: top1_score={top1_score:.3f} < review_threshold={self.review_queue_confidence:.3f}."
            )
            return {
                "action": "reject",
                "category": top1_cat,
                "confidence": float(top1_score),
                "margin": margin,
                "audit": audit_log
            }

        # 4. Check margin ambiguity (Top-1 vs Top-2 separation)
        if margin < self.min_top1_margin:
            audit_log.append(
                f"Ambiguity detected: margin={margin:.3f} < min_margin={self.min_top1_margin:.3f} between '{top1_cat}' and '{top2_cat}'."
            )
            # If top-1 is high but margin is tight, escalate to Category Manager
            if top1_score >= self.auto_tag_confidence:
                action = "escalate"
                audit_log.append("Escalated to Category Manager due to high-confidence category conflict.")
            else:
                action = "human_review"
                audit_log.append("Routed to Human Review queue for disambiguation.")
            return {
                "action": action,
                "category": top1_cat,
                "confidence": float(top1_score),
                "margin": margin,
                "audit": audit_log
            }

        # 5. High-confidence auto-publish check
        if top1_score >= self.auto_tag_confidence:
            audit_log.append(
                f"Auto-publish approved: top1_score={top1_score:.3f} >= {self.auto_tag_confidence:.3f} and margin={margin:.3f} >= {self.min_top1_margin:.3f}."
            )
            return {
                "action": "auto_tag",
                "category": top1_cat,
                "confidence": float(top1_score),
                "margin": margin,
                "audit": audit_log
            }

        # 6. Moderate confidence human verification queue
        audit_log.append(
            f"Human review required: review_threshold={self.review_queue_confidence:.3f} <= score={top1_score:.3f} < auto_threshold={self.auto_tag_confidence:.3f}."
        )
        return {
            "action": "human_review",
            "category": top1_cat,
            "confidence": float(top1_score),
            "margin": margin,
            "audit": audit_log
        }

    def batch_decide(
        self,
        scores_list: List[Dict[str, float]],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Applies decision rules across a batch of SKU predictions."""
        metadata_list = metadata_list or [{}] * len(scores_list)
        return [self.decide(s, m) for s, m in zip(scores_list, metadata_list)]

