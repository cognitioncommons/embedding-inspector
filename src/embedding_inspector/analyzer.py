"""
Embedding analysis functions.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .loader import EmbeddingSet


@dataclass
class SimilarityResult:
    """Result of a similarity search."""
    index: int
    similarity: float
    id: Optional[str] = None
    text: Optional[str] = None


@dataclass
class DistributionStats:
    """Statistics about similarity distribution."""
    mean: float
    std: float
    min: float
    max: float
    median: float
    percentile_25: float
    percentile_75: float
    percentile_95: float


@dataclass
class ClusterResult:
    """Result of clustering."""
    labels: np.ndarray
    n_clusters: int
    centers: Optional[np.ndarray] = None
    sizes: Optional[list[int]] = None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarities(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Compute cosine similarities between query and all vectors."""
    # Normalize query
    query_norm = query / (np.linalg.norm(query) + 1e-10)

    # Normalize all vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / (norms + 1e-10)

    # Compute dot products
    return np.dot(vectors_norm, query_norm)


def find_nearest(
    embeddings: EmbeddingSet,
    query: np.ndarray,
    k: int = 10,
    exclude_self: bool = False,
) -> list[SimilarityResult]:
    """
    Find the k nearest neighbors to a query vector.

    Args:
        embeddings: The embedding set to search
        query: Query vector
        k: Number of neighbors to return
        exclude_self: If True, exclude exact matches

    Returns:
        List of SimilarityResult objects
    """
    if query.ndim == 1:
        query = query.reshape(1, -1)

    similarities = cosine_similarities(query.flatten(), embeddings.vectors)

    # Get top k indices
    if exclude_self:
        # Set similarity of exact matches to -inf
        exact_match_mask = similarities > 0.9999
        similarities = similarities.copy()
        similarities[exact_match_mask] = -np.inf

    top_indices = np.argsort(similarities)[::-1][:k]

    results = []
    for idx in top_indices:
        results.append(SimilarityResult(
            index=int(idx),
            similarity=float(similarities[idx]),
            id=embeddings.get_id(idx),
            text=embeddings.get_text(idx),
        ))

    return results


def compute_similarities(
    embeddings: EmbeddingSet,
    sample_size: Optional[int] = None,
) -> np.ndarray:
    """
    Compute pairwise similarities for a sample of vectors.

    Args:
        embeddings: The embedding set
        sample_size: If provided, use a random sample

    Returns:
        Array of similarity values (flattened upper triangle)
    """
    vectors = embeddings.vectors

    if sample_size and sample_size < len(vectors):
        indices = np.random.choice(len(vectors), sample_size, replace=False)
        vectors = vectors[indices]

    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / (norms + 1e-10)

    # Compute similarity matrix
    sim_matrix = np.dot(vectors_norm, vectors_norm.T)

    # Extract upper triangle (excluding diagonal)
    n = len(vectors)
    triu_indices = np.triu_indices(n, k=1)

    return sim_matrix[triu_indices]


def analyze_distribution(
    embeddings: EmbeddingSet,
    sample_size: int = 1000,
) -> DistributionStats:
    """
    Analyze the similarity distribution of embeddings.

    Args:
        embeddings: The embedding set
        sample_size: Number of pairs to sample

    Returns:
        DistributionStats with summary statistics
    """
    similarities = compute_similarities(embeddings, sample_size)

    return DistributionStats(
        mean=float(np.mean(similarities)),
        std=float(np.std(similarities)),
        min=float(np.min(similarities)),
        max=float(np.max(similarities)),
        median=float(np.median(similarities)),
        percentile_25=float(np.percentile(similarities, 25)),
        percentile_75=float(np.percentile(similarities, 75)),
        percentile_95=float(np.percentile(similarities, 95)),
    )


def cluster_embeddings(
    embeddings: EmbeddingSet,
    n_clusters: int = 10,
    method: str = "kmeans",
) -> ClusterResult:
    """
    Cluster embeddings using specified method.

    Args:
        embeddings: The embedding set
        n_clusters: Number of clusters
        method: Clustering method ('kmeans' or 'agglomerative')

    Returns:
        ClusterResult with labels and cluster info
    """
    try:
        from sklearn.cluster import KMeans, AgglomerativeClustering
    except ImportError:
        raise ImportError("scikit-learn is required for clustering: pip install scikit-learn")

    vectors = embeddings.vectors

    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(vectors)
        centers = clusterer.cluster_centers_
    elif method == "agglomerative":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(vectors)
        centers = None
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Compute cluster sizes
    sizes = [int(np.sum(labels == i)) for i in range(n_clusters)]

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        centers=centers,
        sizes=sizes,
    )


def reduce_dimensions(
    embeddings: EmbeddingSet,
    n_components: int = 2,
    method: str = "pca",
) -> np.ndarray:
    """
    Reduce embedding dimensions for visualization.

    Args:
        embeddings: The embedding set
        n_components: Target dimensionality (2 or 3)
        method: 'pca', 'tsne', or 'umap'

    Returns:
        Reduced vectors of shape (n_samples, n_components)
    """
    vectors = embeddings.vectors

    if method == "pca":
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("scikit-learn is required for PCA: pip install scikit-learn")

        reducer = PCA(n_components=n_components, random_state=42)
        return reducer.fit_transform(vectors)

    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            raise ImportError("scikit-learn is required for t-SNE: pip install scikit-learn")

        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(vectors) - 1))
        return reducer.fit_transform(vectors)

    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn is required for UMAP: pip install umap-learn")

        reducer = umap.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(vectors)

    else:
        raise ValueError(f"Unknown reduction method: {method}")


def find_outliers(
    embeddings: EmbeddingSet,
    threshold: float = 2.0,
) -> list[int]:
    """
    Find vectors that are outliers (far from all others).

    Args:
        embeddings: The embedding set
        threshold: Z-score threshold for outliers

    Returns:
        List of outlier indices
    """
    vectors = embeddings.vectors
    n = len(vectors)

    # Compute average similarity for each vector
    avg_similarities = np.zeros(n)

    for i in range(n):
        sims = cosine_similarities(vectors[i], vectors)
        # Exclude self
        sims[i] = 0
        avg_similarities[i] = np.mean(sims)

    # Find outliers using z-score
    mean = np.mean(avg_similarities)
    std = np.std(avg_similarities)

    z_scores = (avg_similarities - mean) / (std + 1e-10)

    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]

    return outlier_indices.tolist()


def find_duplicates(
    embeddings: EmbeddingSet,
    threshold: float = 0.99,
) -> list[tuple[int, int, float]]:
    """
    Find near-duplicate vectors.

    Args:
        embeddings: The embedding set
        threshold: Similarity threshold for duplicates

    Returns:
        List of (index1, index2, similarity) tuples
    """
    vectors = embeddings.vectors
    n = len(vectors)

    duplicates = []

    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / (norms + 1e-10)

    for i in range(n):
        sims = np.dot(vectors_norm[i], vectors_norm[i + 1:].T)
        high_sim_indices = np.where(sims > threshold)[0] + i + 1

        for j in high_sim_indices:
            duplicates.append((i, int(j), float(sims[j - i - 1])))

    return duplicates
