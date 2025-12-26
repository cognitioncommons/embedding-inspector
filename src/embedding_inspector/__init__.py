"""
embedding-inspector: Explore and debug embedding spaces from the CLI.

Load embeddings from various formats and analyze similarity, clustering,
and distribution patterns.
"""

__version__ = "0.1.0"

from .loader import load_embeddings, EmbeddingSet
from .analyzer import (
    find_nearest,
    compute_similarities,
    analyze_distribution,
    cluster_embeddings,
)

__all__ = [
    "load_embeddings",
    "EmbeddingSet",
    "find_nearest",
    "compute_similarities",
    "analyze_distribution",
    "cluster_embeddings",
]
