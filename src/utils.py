"""Utility functions for CLIP-Catalog: configuration, seeds, device management,
and vector indexing wrappers.
"""

import os
import json
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Set ARM64 / Snapdragon CPU flag check bypass for Polars / NumPy environments if present
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across standard library, numpy, and torch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(preference: str = "auto") -> str:
    """Detects available acceleration hardware (cuda, mps, cpu)."""
    if preference != "auto":
        return preference
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file with pure-Python fallback parser if pyyaml missing."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Robust fallback parser for basic YAML key-value sections
        config: Dict[str, Any] = {}
        current_section = None
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if ":" in stripped:
                    parts = stripped.split(":", 1)
                    key = parts[0].strip()
                    val = parts[1].split("#")[0].strip()
                    if line.startswith("  ") and current_section:
                        if val == "":
                            parsed_val = {}
                        elif val.lower() == "true":
                            parsed_val = True
                        elif val.lower() == "false":
                            parsed_val = False
                        elif val.startswith("[") and val.endswith("]"):
                            parsed_val = [x.strip() for x in val[1:-1].split(",") if x.strip()]
                        else:
                            try:
                                parsed_val = float(val) if "." in val or "e" in val.lower() else int(val)
                            except ValueError:
                                parsed_val = val
                        config[current_section][key] = parsed_val
                    else:
                        current_section = key
                        config[current_section] = {}
        return config


def l2_normalize(vectors: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalizes an array of vectors along specified axis."""
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / np.maximum(norms, eps)


def save_jsonl(records: List[Dict[str, Any]], filepath: str) -> None:
    """Writes a list of dictionaries to a JSON Lines file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Reads records from a JSON Lines file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSONL file not found: {filepath}")
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class NumpyIndexFlatIP:
    """Pure-NumPy Inner Product search index mirroring FAISS IndexFlatIP API.

    Used automatically when faiss is not installed or for lightweight environments.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: Optional[np.ndarray] = None
        self.ntotal: int = 0

    def add(self, x: np.ndarray) -> None:
        """Adds normalized vectors of shape (N, dimension) to index."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.vectors is None:
            self.vectors = x
        else:
            self.vectors = np.vstack([self.vectors, x])
        self.ntotal = self.vectors.shape[0]

    def search(self, query: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Performs inner product search. Returns (distances/scores, indices)."""
        if self.vectors is None or self.ntotal == 0:
            raise ValueError("Index is empty. Add vectors before searching.")
        query = np.ascontiguousarray(query, dtype=np.float32)
        scores = np.dot(query, self.vectors.T)  # (Q, N)
        # Find top_k indices for each query
        top_k = min(top_k, self.ntotal)
        indices = np.argsort(-scores, axis=-1)[:, :top_k]
        distances = np.take_along_axis(scores, indices, axis=-1)
        return distances, indices

    def reset(self) -> None:
        """Clears indexed vectors."""
        self.vectors = None
        self.ntotal = 0


def create_vector_index(dimension: int, index_type: str = "IndexFlatIP"):
    """Factory creating FAISS index or falling back to NumPyIndexFlatIP."""
    try:
        import faiss
        if index_type == "IndexFlatIP":
            return faiss.IndexFlatIP(dimension)
        elif index_type == "IndexFlatL2":
            return faiss.IndexFlatL2(dimension)
        else:
            return faiss.IndexFlatIP(dimension)
    except ImportError:
        return NumpyIndexFlatIP(dimension)
