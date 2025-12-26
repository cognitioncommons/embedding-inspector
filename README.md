# embedding-inspector

Explore and debug embedding spaces from the CLI.

Load embeddings from various formats (numpy, pickle, JSON, FAISS) and analyze similarity, clustering, and distribution patterns.

## Installation

```bash
pip install embedding-inspector
```

For full functionality (clustering, visualization):

```bash
pip install embedding-inspector[all]
```

Or from source:

```bash
git clone https://github.com/cognitioncommons/embedding-inspector.git
cd embedding-inspector
pip install -e ".[all]"
```

## Usage

### View Embedding Info

```bash
embedding-inspector info vectors.npy
```

### Search Nearest Neighbors

```bash
# Find 10 nearest neighbors to vector at index 0
embedding-inspector search vectors.npy --index 0 -k 10

# Exclude exact matches
embedding-inspector search vectors.npy --index 0 --exclude-self
```

### Analyze Distribution

```bash
# Get similarity distribution statistics
embedding-inspector stats vectors.npy

# With larger sample
embedding-inspector stats vectors.npy --sample 5000
```

### Cluster Embeddings

```bash
# K-means clustering into 10 clusters
embedding-inspector cluster vectors.npy --n 10

# Agglomerative clustering
embedding-inspector cluster vectors.npy --n 10 --method agglomerative
```

### Find Duplicates

```bash
# Find near-duplicate vectors (>0.99 similarity)
embedding-inspector duplicates vectors.npy

# Custom threshold
embedding-inspector duplicates vectors.npy --threshold 0.95
```

### Find Outliers

```bash
# Find outlier vectors
embedding-inspector outliers vectors.npy

# Custom z-score threshold
embedding-inspector outliers vectors.npy --threshold 3.0
```

### Visualize Embeddings

```bash
# PCA visualization
embedding-inspector visualize vectors.npy --method pca -o viz.png

# t-SNE (slower but better for clusters)
embedding-inspector visualize vectors.npy --method tsne -o viz.png

# UMAP (requires umap-learn)
embedding-inspector visualize vectors.npy --method umap -o viz.png
```

### Sample Vectors

```bash
# Show first 20 vectors with metadata
embedding-inspector sample vectors.npy

# Start from index 100
embedding-inspector sample vectors.npy --start 100 --limit 50
```

## Supported File Formats

### NumPy (.npy, .npz)

```python
import numpy as np

# Simple array
vectors = np.random.randn(1000, 384)
np.save("vectors.npy", vectors)

# With metadata
np.savez("vectors.npz",
    vectors=vectors,
    ids=["id_0", "id_1", ...],
    texts=["text 0", "text 1", ...]
)
```

### Pickle (.pkl)

```python
import pickle

data = {
    "vectors": vectors,
    "ids": ["id_0", "id_1", ...],
    "texts": ["text 0", "text 1", ...]
}

with open("vectors.pkl", "wb") as f:
    pickle.dump(data, f)
```

### JSON (.json)

```json
{
  "vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "ids": ["id_0", "id_1"],
  "texts": ["text 0", "text 1"]
}
```

Or as a list of objects:

```json
[
  {"id": "doc_1", "text": "Hello world", "embedding": [0.1, 0.2, ...]},
  {"id": "doc_2", "text": "Goodbye", "embedding": [0.3, 0.4, ...]}
]
```

### JSONL (.jsonl)

```jsonl
{"id": "doc_1", "text": "Hello world", "embedding": [0.1, 0.2, ...]}
{"id": "doc_2", "text": "Goodbye", "embedding": [0.3, 0.4, ...]}
```

### CSV (.csv)

```csv
id,text,dim_0,dim_1,dim_2,...
doc_1,Hello world,0.1,0.2,0.3,...
doc_2,Goodbye,0.3,0.4,0.5,...
```

### FAISS (.faiss)

Requires `faiss-cpu`:

```python
import faiss

index = faiss.IndexFlatL2(384)
index.add(vectors)
faiss.write_index(index, "vectors.faiss")
```

## Python API

```python
from embedding_inspector import load_embeddings, find_nearest, analyze_distribution

# Load embeddings
embeddings = load_embeddings("vectors.npy")
print(f"Loaded {embeddings.n_vectors} vectors of dimension {embeddings.dimensions}")

# Find nearest neighbors
query = embeddings.get_vector(0)
neighbors = find_nearest(embeddings, query, k=5)
for n in neighbors:
    print(f"Index {n.index}: similarity {n.similarity:.4f}")

# Analyze distribution
stats = analyze_distribution(embeddings, sample_size=1000)
print(f"Mean similarity: {stats.mean:.4f}")
print(f"Std deviation: {stats.std:.4f}")
```

### Clustering

```python
from embedding_inspector import cluster_embeddings

result = cluster_embeddings(embeddings, n_clusters=10, method="kmeans")
print(f"Cluster sizes: {result.sizes}")

# Get vectors in cluster 0
cluster_0_indices = [i for i, label in enumerate(result.labels) if label == 0]
```

### Dimensionality Reduction

```python
from embedding_inspector.analyzer import reduce_dimensions

# For visualization
reduced = reduce_dimensions(embeddings, n_components=2, method="pca")
# reduced.shape = (n_vectors, 2)
```

## Example Output

### Stats Command

```
╭───────── Embedding Statistics ──────────╮
│ Vectors: 10,000                         │
│ Dimensions: 384                         │
│ Sample Size: 1,000                      │
╰─────────────────────────────────────────╯

Similarity Distribution
Statistic          Value
Mean               0.1234
Std Dev            0.0567
Min               -0.2345
Max                0.8901
Median             0.1100
25th Percentile    0.0823
75th Percentile    0.1645
95th Percentile    0.2301

Normal similarity distribution
```

### Cluster Command

```
Clustering Results (kmeans)
10,000 vectors into 10 clusters

Cluster  Size    Percent  Sample Texts
0        1,245   12.5%    machine learning | deep... | neural...
1        892     8.9%     database query | SQL... | postgres...
2        1,567   15.7%    web development | React... | frontend...
...
```

## Optional Dependencies

| Feature | Package | Install |
|---------|---------|---------|
| Clustering | scikit-learn | `pip install scikit-learn` |
| t-SNE | scikit-learn | `pip install scikit-learn` |
| UMAP | umap-learn | `pip install umap-learn` |
| FAISS | faiss-cpu | `pip install faiss-cpu` |
| Visualization | matplotlib | `pip install matplotlib` |

Install all: `pip install embedding-inspector[all]`

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

Part of the [Cognition Commons](https://cognitioncommons.org) project.
