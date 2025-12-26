"""
Load embeddings from various file formats.
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import numpy as np


@dataclass
class EmbeddingSet:
    """A set of embeddings with optional metadata."""
    vectors: np.ndarray  # Shape: (n_samples, dimensions)
    ids: Optional[list[str]] = None  # Optional IDs for each vector
    texts: Optional[list[str]] = None  # Optional source texts
    metadata: dict = field(default_factory=dict)

    @property
    def n_vectors(self) -> int:
        """Number of vectors in the set."""
        return self.vectors.shape[0]

    @property
    def dimensions(self) -> int:
        """Dimensionality of vectors."""
        return self.vectors.shape[1]

    def get_vector(self, index: int) -> np.ndarray:
        """Get a single vector by index."""
        return self.vectors[index]

    def get_text(self, index: int) -> Optional[str]:
        """Get the source text for a vector."""
        if self.texts and index < len(self.texts):
            return self.texts[index]
        return None

    def get_id(self, index: int) -> str:
        """Get the ID for a vector."""
        if self.ids and index < len(self.ids):
            return self.ids[index]
        return str(index)

    def subset(self, indices: list[int]) -> "EmbeddingSet":
        """Create a subset with selected indices."""
        return EmbeddingSet(
            vectors=self.vectors[indices],
            ids=[self.ids[i] for i in indices] if self.ids else None,
            texts=[self.texts[i] for i in indices] if self.texts else None,
            metadata=self.metadata.copy(),
        )


def load_embeddings(source: str | Path) -> EmbeddingSet:
    """
    Load embeddings from various file formats.

    Supported formats:
    - .npy: NumPy array file
    - .npz: NumPy compressed archive
    - .pkl/.pickle: Pickle file (dict or array)
    - .json/.jsonl: JSON with embeddings
    - .csv: CSV with numeric columns
    - .faiss: FAISS index (requires faiss-cpu)

    Args:
        source: Path to embedding file

    Returns:
        EmbeddingSet with loaded vectors
    """
    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        return load_numpy(path)
    elif suffix == ".npz":
        return load_numpy_compressed(path)
    elif suffix in (".pkl", ".pickle"):
        return load_pickle(path)
    elif suffix == ".json":
        return load_json(path)
    elif suffix == ".jsonl":
        return load_jsonl(path)
    elif suffix == ".csv":
        return load_csv(path)
    elif suffix == ".faiss":
        return load_faiss(path)
    else:
        # Try to detect format from content
        return load_auto(path)


def load_numpy(path: Path) -> EmbeddingSet:
    """Load from .npy file."""
    vectors = np.load(path)

    if vectors.ndim == 1:
        # Single vector, reshape to (1, dim)
        vectors = vectors.reshape(1, -1)

    return EmbeddingSet(
        vectors=vectors.astype(np.float32),
        metadata={"source": str(path), "format": "numpy"},
    )


def load_numpy_compressed(path: Path) -> EmbeddingSet:
    """Load from .npz file."""
    data = np.load(path)

    # Look for common key names
    for key in ["vectors", "embeddings", "data", "arr_0"]:
        if key in data:
            vectors = data[key]
            break
    else:
        # Use first array
        vectors = data[list(data.keys())[0]]

    # Look for metadata
    ids = None
    texts = None

    if "ids" in data:
        ids = data["ids"].tolist()
    if "texts" in data:
        texts = data["texts"].tolist()

    return EmbeddingSet(
        vectors=vectors.astype(np.float32),
        ids=ids,
        texts=texts,
        metadata={"source": str(path), "format": "numpy_compressed"},
    )


def load_pickle(path: Path) -> EmbeddingSet:
    """Load from pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, np.ndarray):
        return EmbeddingSet(
            vectors=data.astype(np.float32),
            metadata={"source": str(path), "format": "pickle"},
        )
    elif isinstance(data, dict):
        # Look for vectors
        vectors = None
        for key in ["vectors", "embeddings", "data"]:
            if key in data and isinstance(data[key], (np.ndarray, list)):
                vectors = np.array(data[key])
                break

        if vectors is None:
            raise ValueError("No vectors found in pickle file")

        return EmbeddingSet(
            vectors=vectors.astype(np.float32),
            ids=data.get("ids"),
            texts=data.get("texts"),
            metadata={"source": str(path), "format": "pickle"},
        )
    elif isinstance(data, list):
        # List of vectors or dicts
        if isinstance(data[0], (list, np.ndarray)):
            vectors = np.array(data)
        elif isinstance(data[0], dict):
            vectors = np.array([d.get("embedding", d.get("vector")) for d in data])
            texts = [d.get("text", d.get("content")) for d in data]
            ids = [d.get("id", str(i)) for i, d in enumerate(data)]
            return EmbeddingSet(
                vectors=vectors.astype(np.float32),
                ids=ids,
                texts=texts,
                metadata={"source": str(path), "format": "pickle"},
            )
        else:
            raise ValueError("Unexpected pickle format")

        return EmbeddingSet(
            vectors=vectors.astype(np.float32),
            metadata={"source": str(path), "format": "pickle"},
        )
    else:
        raise ValueError(f"Unexpected pickle type: {type(data)}")


def load_json(path: Path) -> EmbeddingSet:
    """Load from JSON file."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Look for vectors
        vectors = None
        for key in ["vectors", "embeddings", "data"]:
            if key in data and isinstance(data[key], list):
                vectors = np.array(data[key])
                break

        if vectors is None:
            raise ValueError("No vectors found in JSON file")

        return EmbeddingSet(
            vectors=vectors.astype(np.float32),
            ids=data.get("ids"),
            texts=data.get("texts"),
            metadata={"source": str(path), "format": "json"},
        )
    elif isinstance(data, list):
        # List of vectors or objects
        if isinstance(data[0], list):
            vectors = np.array(data)
            return EmbeddingSet(
                vectors=vectors.astype(np.float32),
                metadata={"source": str(path), "format": "json"},
            )
        elif isinstance(data[0], dict):
            vectors = []
            ids = []
            texts = []

            for item in data:
                vec = item.get("embedding") or item.get("vector") or item.get("values")
                vectors.append(vec)
                ids.append(item.get("id", str(len(ids))))
                texts.append(item.get("text") or item.get("content"))

            return EmbeddingSet(
                vectors=np.array(vectors, dtype=np.float32),
                ids=ids,
                texts=texts if any(texts) else None,
                metadata={"source": str(path), "format": "json"},
            )
    else:
        raise ValueError(f"Unexpected JSON format: {type(data)}")


def load_jsonl(path: Path) -> EmbeddingSet:
    """Load from JSONL file (one JSON object per line)."""
    vectors = []
    ids = []
    texts = []

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            vec = item.get("embedding") or item.get("vector") or item.get("values")
            vectors.append(vec)
            ids.append(item.get("id", str(len(ids))))
            texts.append(item.get("text") or item.get("content"))

    return EmbeddingSet(
        vectors=np.array(vectors, dtype=np.float32),
        ids=ids,
        texts=texts if any(texts) else None,
        metadata={"source": str(path), "format": "jsonl"},
    )


def load_csv(path: Path) -> EmbeddingSet:
    """Load from CSV file."""
    import csv

    vectors = []
    ids = []
    texts = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # Detect which columns are embeddings
        vector_cols = [h for h in headers if h.startswith("dim_") or h.startswith("v")]

        if not vector_cols:
            # All numeric columns except id/text
            non_vector = {"id", "text", "content", "label"}
            vector_cols = [h for h in headers if h.lower() not in non_vector]

        for row in reader:
            vec = [float(row[c]) for c in vector_cols]
            vectors.append(vec)
            ids.append(row.get("id", str(len(ids))))
            texts.append(row.get("text") or row.get("content"))

    return EmbeddingSet(
        vectors=np.array(vectors, dtype=np.float32),
        ids=ids,
        texts=texts if any(texts) else None,
        metadata={"source": str(path), "format": "csv"},
    )


def load_faiss(path: Path) -> EmbeddingSet:
    """Load from FAISS index file."""
    try:
        import faiss
    except ImportError:
        raise ImportError("faiss-cpu is required to load .faiss files: pip install faiss-cpu")

    index = faiss.read_index(str(path))

    # Extract vectors from index
    if hasattr(index, "reconstruct_n"):
        n = index.ntotal
        d = index.d
        vectors = np.zeros((n, d), dtype=np.float32)
        index.reconstruct_n(0, n, vectors)
    else:
        raise ValueError("Cannot extract vectors from this FAISS index type")

    return EmbeddingSet(
        vectors=vectors,
        metadata={"source": str(path), "format": "faiss", "index_type": type(index).__name__},
    )


def load_auto(path: Path) -> EmbeddingSet:
    """Try to auto-detect format."""
    # Try numpy first
    try:
        return load_numpy(path)
    except Exception:
        pass

    # Try pickle
    try:
        return load_pickle(path)
    except Exception:
        pass

    # Try JSON
    try:
        return load_json(path)
    except Exception:
        pass

    raise ValueError(f"Could not detect format of: {path}")
