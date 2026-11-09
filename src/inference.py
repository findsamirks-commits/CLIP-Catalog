"""Inference engine for CLIP-Catalog: zero-shot classification and FAISS retrieval.

Provides the four core operational modes:
1. Zero-shot classification with prompt ensembling
2. Text -> SKU natural language semantic search
3. Image -> SKU visual similarity search (competitor match / substitutes)
4. Image -> Category open-vocabulary tagging
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import l2_normalize, load_config
from src.model import CLIPCatalogModel
from src.index import CatalogIndex, build_index
from src.data import generate_synthetic_catalog


class CLIPRetriever:
    """High-level retrieval and zero-shot categorization interface."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        index_path: str = "data/processed/catalog.faiss",
        config_path: str = "configs/clip_catalog.yaml",
        temperature: float = 0.07
    ):
        self.checkpoint_path = checkpoint_path
        self.index_path = index_path
        self.temperature = temperature

        # Load config if present
        cfg = load_config(config_path) if os.path.exists(config_path) else {}
        self.model_cfg = cfg.get("model", {})
        self.retrieval_cfg = cfg.get("retrieval", {})

        # Initialize model
        self.model = CLIPCatalogModel(
            backbone=self.model_cfg.get("backbone", "ViT-B-32"),
            pretrained=self.model_cfg.get("pretrained", "laion2b_s34b_b79k"),
            temperature=self.temperature
        )

        # Initialize or load catalog retrieval index
        self.index: Optional[CatalogIndex] = None
        self._ensure_index_loaded()

    def _ensure_index_loaded(self) -> None:
        """Loads index or lazily compiles a catalog index from synthetic data."""
        meta_path = self.index_path.rsplit(".", 1)[0] + "_meta.json"
        if os.path.exists(self.index_path) and os.path.exists(meta_path):
            try:
                self.index = CatalogIndex.load(self.index_path, meta_path)
                return
            except Exception:
                pass

        # Auto-create lightweight index
        records = generate_synthetic_catalog(num_skus=100)
        self.index = CatalogIndex(dimension=512)
        self.index.build_from_records(records, self.model)

    def zero_shot_classify(
        self,
        image: Any,
        candidate_labels: List[str],
        prompt_template: Union[str, List[str]] = "a photo of {}"
    ) -> Dict[str, float]:
        """Classifies an image across candidate labels using zero-shot prompt ensembling.

        Args:
            image: PIL Image, image array, or filepath.
            candidate_labels: Candidate taxonomy category strings.
            prompt_template: Template string or list of template strings for ensembling.

        Returns:
            Dictionary mapping label to normalized softmax probability, sorted descending.
        """
        if not candidate_labels:
            return {}

        templates = [prompt_template] if isinstance(prompt_template, str) else prompt_template

        # Encode image
        image_embedding = self.model.encode_image(image)  # (1, dim)
        image_embedding = l2_normalize(image_embedding, axis=-1)

        # Compute ensemble text embeddings for each candidate label
        label_embeddings: List[np.ndarray] = []
        for label in candidate_labels:
            prompts = [tmpl.format(label) for tmpl in templates]
            prompt_vecs = self.model.encode_text(prompts)  # (num_templates, dim)
            # Ensemble average and re-normalize
            mean_vec = np.mean(prompt_vecs, axis=0, keepdims=True)
            norm_vec = l2_normalize(mean_vec, axis=-1)
            label_embeddings.append(norm_vec[0])

        label_embeddings_arr = np.array(label_embeddings)  # (num_labels, dim)

        # Compute cosine similarity logits
        logits = np.dot(image_embedding, label_embeddings_arr.T)[0] / self.temperature

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        score_dict = {label: float(prob) for label, prob in zip(candidate_labels, probs)}
        return dict(sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True))

    def text_to_sku(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves top-K catalog SKUs matching a natural-language search query."""
        if self.index is None:
            self._ensure_index_loaded()

        query_embedding = self.model.encode_text(query)
        return self.index.search(query_embedding, top_k=top_k)

    def image_to_sku(self, query_image: Any, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves top-K visually matching catalog SKUs for visual similarity / substitute discovery."""
        if self.index is None:
            self._ensure_index_loaded()

        image_embedding = self.model.encode_image(query_image)
        return self.index.search(image_embedding, top_k=top_k)
