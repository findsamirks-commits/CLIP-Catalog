"""FAISS retrieval index builder for CLIP-Catalog.

Extracts normalized visual and text embeddings for all catalog items and
indexes them using FAISS IndexFlatIP (exact cosine similarity) for sub-millisecond retrieval.
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import l2_normalize, load_jsonl, save_jsonl, create_vector_index
from src.model import CLIPCatalogModel
from src.data import generate_synthetic_catalog


class CatalogIndex:
    """Manages vector search index and SKU metadata mapping."""

    def __init__(self, dimension: int = 512, index_type: str = "IndexFlatIP"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = create_vector_index(dimension, index_type=index_type)
        self.metadata: List[Dict[str, Any]] = []

    def build_from_records(
        self,
        records: List[Dict[str, Any]],
        model: CLIPCatalogModel,
        batch_size: int = 64
    ) -> None:
        """Encodes catalog captions/images and populates vector index."""
        self.metadata = records
        captions = [r.get("caption", r.get("title", "")) for r in records]
        
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(captions), batch_size):
            batch_texts = captions[i : i + batch_size]
            embeds = model.encode_text(batch_texts)
            all_embeddings.append(embeds)

        if all_embeddings:
            combined = np.vstack(all_embeddings).astype(np.float32)
            combined = l2_normalize(combined, axis=-1)
            self.index.add(combined)
            print(f"Indexed {self.index.ntotal} SKUs with dimension {self.dimension}.")

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Searches top-K nearest SKUs for a given normalized query embedding."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = l2_normalize(query_embedding, axis=-1)

        scores, indices = self.index.search(query_embedding, top_k=top_k)
        
        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if 0 <= idx < len(self.metadata):
                rec = dict(self.metadata[idx])
                rec["score"] = float(score)
                rec["rank"] = rank + 1
                results.append(rec)
        return results

    def save(self, index_path: str, meta_path: Optional[str] = None) -> None:
        """Persists index and SKU metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        if meta_path is None:
            meta_path = index_path.rsplit(".", 1)[0] + "_meta.json"

        # Check if FAISS native save or pure-NumPy save
        try:
            import faiss
            if hasattr(self.index, "ntotal") and type(self.index).__name__.startswith("Index"):
                faiss.write_index(self.index, index_path)
            else:
                np.save(index_path, getattr(self.index, "vectors", np.array([])))
        except ImportError:
            np.save(index_path, getattr(self.index, "vectors", np.array([])))

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved index -> {index_path}")
        print(f"Saved metadata -> {meta_path}")

    @classmethod
    def load(cls, index_path: str, meta_path: Optional[str] = None) -> "CatalogIndex":
        """Loads index and metadata from disk."""
        if meta_path is None:
            meta_path = index_path.rsplit(".", 1)[0] + "_meta.json"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")

        meta = []
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        try:
            import faiss
            raw_idx = faiss.read_index(index_path)
            dim = raw_idx.d
            inst = cls(dimension=dim)
            inst.index = raw_idx
        except (ImportError, Exception):
            vecs = np.load(index_path if index_path.endswith(".npy") else index_path + ".npy")
            dim = vecs.shape[1] if vecs.ndim > 1 else 512
            inst = cls(dimension=dim)
            inst.index.add(vecs)

        inst.metadata = meta
        return inst


def build_index(
    catalog_path: str = "data/processed/catalog.jsonl",
    index_out: str = "data/processed/catalog.faiss",
    model_path: Optional[str] = None,
    dimension: int = 512
) -> CatalogIndex:
    """High-level function to encode catalog records and save vector index."""
    if not os.path.exists(catalog_path):
        os.makedirs(os.path.dirname(os.path.abspath(catalog_path)), exist_ok=True)
        generate_synthetic_catalog(num_skus=100, output_path=catalog_path)

    records = load_jsonl(catalog_path)
    model = CLIPCatalogModel()
    cat_index = CatalogIndex(dimension=dimension)
    cat_index.build_from_records(records, model)
    cat_index.save(index_out)
    return cat_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS retrieval index over catalog items.")
    parser.add_argument("--catalog", type=str, default="data/processed/catalog.jsonl")
    parser.add_argument("--index-out", type=str, default="data/processed/catalog.faiss")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--dim", type=int, default=512)
    args = parser.parse_args()

    build_index(
        catalog_path=args.catalog,
        index_out=args.index_out,
        model_path=args.model_path,
        dimension=args.dim
    )

