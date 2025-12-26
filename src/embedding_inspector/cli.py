"""
Command-line interface for embedding-inspector.
"""

import json
from pathlib import Path
from typing import Optional

import click
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

from .loader import load_embeddings, EmbeddingSet
from .analyzer import (
    find_nearest,
    analyze_distribution,
    cluster_embeddings,
    find_outliers,
    find_duplicates,
    reduce_dimensions,
)

console = Console()


@click.group()
def main():
    """
    Explore and debug embedding spaces from the CLI.

    Load embeddings from various formats (numpy, pickle, JSON, FAISS)
    and analyze similarity, clustering, and distribution patterns.

    Examples:

        embedding-inspector info vectors.npy

        embedding-inspector search vectors.npy --index 0 -k 5

        embedding-inspector stats vectors.npy

        embedding-inspector cluster vectors.npy --n 10

        embedding-inspector duplicates vectors.npy
    """
    pass


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def info(file: str, as_json: bool):
    """
    Show information about an embedding file.
    """
    embeddings = load_embeddings(file)

    if as_json:
        output = {
            "file": file,
            "format": embeddings.metadata.get("format", "unknown"),
            "n_vectors": embeddings.n_vectors,
            "dimensions": embeddings.dimensions,
            "has_ids": embeddings.ids is not None,
            "has_texts": embeddings.texts is not None,
        }
        console.print_json(json.dumps(output))
        return

    console.print()
    console.print(Panel(
        f"[bold]File:[/bold] {file}\n"
        f"[bold]Format:[/bold] {embeddings.metadata.get('format', 'unknown')}\n"
        f"[bold]Vectors:[/bold] {embeddings.n_vectors:,}\n"
        f"[bold]Dimensions:[/bold] {embeddings.dimensions}\n"
        f"[bold]Has IDs:[/bold] {'Yes' if embeddings.ids else 'No'}\n"
        f"[bold]Has Texts:[/bold] {'Yes' if embeddings.texts else 'No'}",
        title="Embedding Set Info",
        border_style="blue",
    ))


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--index", "-i", type=int, required=True, help="Index of query vector")
@click.option("--k", "-k", default=10, help="Number of neighbors")
@click.option("--exclude-self", is_flag=True, help="Exclude exact matches")
def search(file: str, index: int, k: int, exclude_self: bool):
    """
    Find nearest neighbors to a vector by index.
    """
    embeddings = load_embeddings(file)

    if index < 0 or index >= embeddings.n_vectors:
        console.print(f"[red]Index {index} out of range (0-{embeddings.n_vectors - 1})[/red]")
        return

    query = embeddings.get_vector(index)
    results = find_nearest(embeddings, query, k=k, exclude_self=exclude_self)

    console.print()
    console.print(f"[bold]Nearest neighbors to vector {index}[/bold]")
    if embeddings.get_text(index):
        console.print(f"[dim]Query: {embeddings.get_text(index)[:80]}...[/dim]")
    console.print()

    table = Table()
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Index", justify="right")
    table.add_column("Similarity", justify="right")
    table.add_column("ID", style="cyan")
    table.add_column("Text", max_width=50)

    for rank, result in enumerate(results, 1):
        text = result.text[:50] + "..." if result.text and len(result.text) > 50 else result.text or ""
        sim_color = "green" if result.similarity > 0.9 else "yellow" if result.similarity > 0.7 else "white"
        table.add_row(
            str(rank),
            str(result.index),
            f"[{sim_color}]{result.similarity:.4f}[/{sim_color}]",
            result.id or "-",
            text,
        )

    console.print(table)


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--sample", "-s", default=1000, help="Sample size for analysis")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def stats(file: str, sample: int, as_json: bool):
    """
    Analyze similarity distribution statistics.
    """
    embeddings = load_embeddings(file)

    with Progress(transient=True) as progress:
        progress.add_task("Analyzing...", total=None)
        dist = analyze_distribution(embeddings, sample_size=sample)

    if as_json:
        output = {
            "n_vectors": embeddings.n_vectors,
            "dimensions": embeddings.dimensions,
            "sample_size": min(sample, embeddings.n_vectors),
            "similarity": {
                "mean": dist.mean,
                "std": dist.std,
                "min": dist.min,
                "max": dist.max,
                "median": dist.median,
                "p25": dist.percentile_25,
                "p75": dist.percentile_75,
                "p95": dist.percentile_95,
            }
        }
        console.print_json(json.dumps(output))
        return

    console.print()
    console.print(Panel(
        f"[bold]Vectors:[/bold] {embeddings.n_vectors:,}\n"
        f"[bold]Dimensions:[/bold] {embeddings.dimensions}\n"
        f"[bold]Sample Size:[/bold] {min(sample, embeddings.n_vectors):,}",
        title="Embedding Statistics",
        border_style="blue",
    ))

    console.print()
    console.print("[bold]Similarity Distribution[/bold]")

    table = Table(box=None)
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Mean", f"{dist.mean:.4f}")
    table.add_row("Std Dev", f"{dist.std:.4f}")
    table.add_row("Min", f"{dist.min:.4f}")
    table.add_row("Max", f"{dist.max:.4f}")
    table.add_row("Median", f"{dist.median:.4f}")
    table.add_row("25th Percentile", f"{dist.percentile_25:.4f}")
    table.add_row("75th Percentile", f"{dist.percentile_75:.4f}")
    table.add_row("95th Percentile", f"{dist.percentile_95:.4f}")

    console.print(table)

    # Interpretation
    console.print()
    if dist.mean > 0.7:
        console.print("[yellow]High average similarity - embeddings may lack diversity[/yellow]")
    elif dist.mean < 0.2:
        console.print("[green]Low average similarity - good diversity[/green]")
    else:
        console.print("[green]Normal similarity distribution[/green]")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--n", "-n", default=10, help="Number of clusters")
@click.option("--method", "-m", type=click.Choice(["kmeans", "agglomerative"]), default="kmeans")
@click.option("--show-samples", "-s", default=3, help="Samples to show per cluster")
def cluster(file: str, n: int, method: str, show_samples: int):
    """
    Cluster embeddings and show distribution.
    """
    embeddings = load_embeddings(file)

    with Progress(transient=True) as progress:
        progress.add_task("Clustering...", total=None)
        result = cluster_embeddings(embeddings, n_clusters=n, method=method)

    console.print()
    console.print(f"[bold]Clustering Results ({method})[/bold]")
    console.print(f"[dim]{embeddings.n_vectors:,} vectors into {n} clusters[/dim]")
    console.print()

    table = Table()
    table.add_column("Cluster", style="cyan", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Percent", justify="right")
    table.add_column("Sample Texts", max_width=60)

    for i in range(n):
        size = result.sizes[i]
        pct = (size / embeddings.n_vectors) * 100

        # Get sample texts
        cluster_indices = np.where(result.labels == i)[0]
        samples = []
        for idx in cluster_indices[:show_samples]:
            text = embeddings.get_text(int(idx))
            if text:
                samples.append(text[:30] + "..." if len(text) > 30 else text)

        sample_text = " | ".join(samples) if samples else "[dim]no texts[/dim]"

        table.add_row(
            str(i),
            f"{size:,}",
            f"{pct:.1f}%",
            sample_text,
        )

    console.print(table)


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--threshold", "-t", default=0.99, help="Similarity threshold")
@click.option("--limit", "-n", default=20, help="Maximum duplicates to show")
def duplicates(file: str, threshold: float, limit: int):
    """
    Find near-duplicate vectors.
    """
    embeddings = load_embeddings(file)

    with Progress(transient=True) as progress:
        progress.add_task("Finding duplicates...", total=None)
        dups = find_duplicates(embeddings, threshold=threshold)

    console.print()
    if not dups:
        console.print(f"[green]No duplicates found with threshold {threshold}[/green]")
        return

    console.print(f"[yellow]Found {len(dups)} duplicate pairs (threshold: {threshold})[/yellow]")
    console.print()

    table = Table()
    table.add_column("Index 1", justify="right")
    table.add_column("Index 2", justify="right")
    table.add_column("Similarity", justify="right")
    table.add_column("Text 1", max_width=30)
    table.add_column("Text 2", max_width=30)

    for i1, i2, sim in dups[:limit]:
        text1 = embeddings.get_text(i1)
        text2 = embeddings.get_text(i2)
        text1 = text1[:30] + "..." if text1 and len(text1) > 30 else text1 or "-"
        text2 = text2[:30] + "..." if text2 and len(text2) > 30 else text2 or "-"

        table.add_row(
            str(i1),
            str(i2),
            f"{sim:.4f}",
            text1,
            text2,
        )

    console.print(table)

    if len(dups) > limit:
        console.print(f"[dim]... and {len(dups) - limit} more[/dim]")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--threshold", "-t", default=2.0, help="Z-score threshold")
@click.option("--limit", "-n", default=20, help="Maximum outliers to show")
def outliers(file: str, threshold: float, limit: int):
    """
    Find outlier vectors.
    """
    embeddings = load_embeddings(file)

    with Progress(transient=True) as progress:
        progress.add_task("Finding outliers...", total=None)
        outlier_indices = find_outliers(embeddings, threshold=threshold)

    console.print()
    if not outlier_indices:
        console.print(f"[green]No outliers found with threshold {threshold}[/green]")
        return

    console.print(f"[yellow]Found {len(outlier_indices)} outliers (z-score > {threshold})[/yellow]")
    console.print()

    table = Table()
    table.add_column("Index", justify="right")
    table.add_column("ID", style="cyan")
    table.add_column("Text", max_width=60)

    for idx in outlier_indices[:limit]:
        text = embeddings.get_text(idx)
        text = text[:60] + "..." if text and len(text) > 60 else text or "-"

        table.add_row(
            str(idx),
            embeddings.get_id(idx) or "-",
            text,
        )

    console.print(table)

    if len(outlier_indices) > limit:
        console.print(f"[dim]... and {len(outlier_indices) - limit} more[/dim]")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--method", "-m", type=click.Choice(["pca", "tsne", "umap"]), default="pca")
@click.option("--output", "-o", type=click.Path(), help="Output image file")
@click.option("--labels", "-l", type=click.Path(exists=True), help="Cluster labels file")
def visualize(file: str, method: str, output: Optional[str], labels: Optional[str]):
    """
    Visualize embeddings in 2D.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[red]matplotlib is required for visualization: pip install matplotlib[/red]")
        return

    embeddings = load_embeddings(file)

    with Progress(transient=True) as progress:
        progress.add_task(f"Reducing dimensions with {method.upper()}...", total=None)
        reduced = reduce_dimensions(embeddings, n_components=2, method=method)

    # Load labels if provided
    cluster_labels = None
    if labels:
        cluster_labels = np.load(labels)

    # Create plot
    plt.figure(figsize=(10, 8))

    if cluster_labels is not None:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, cmap="tab10", alpha=0.6, s=10)
        plt.colorbar(scatter, label="Cluster")
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=10)

    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    plt.title(f"Embedding Visualization ({method.upper()})")
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        console.print(f"Saved visualization to {output}")
    else:
        output = "/tmp/embeddings_viz.png"
        plt.savefig(output, dpi=150)
        console.print(f"Saved visualization to {output}")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--start", "-s", default=0, help="Starting index")
@click.option("--limit", "-n", default=20, help="Number of vectors to show")
def sample(file: str, start: int, limit: int):
    """
    Show a sample of vectors with their metadata.
    """
    embeddings = load_embeddings(file)

    console.print()
    console.print(f"[bold]Sample from {file}[/bold]")
    console.print(f"[dim]Showing {start} to {start + limit - 1} of {embeddings.n_vectors}[/dim]")
    console.print()

    table = Table()
    table.add_column("Index", justify="right", style="dim")
    table.add_column("ID", style="cyan")
    table.add_column("Vector (first 5 dims)", max_width=40)
    table.add_column("Text", max_width=40)

    end = min(start + limit, embeddings.n_vectors)
    for i in range(start, end):
        vec = embeddings.get_vector(i)
        vec_str = ", ".join(f"{v:.3f}" for v in vec[:5]) + "..."

        text = embeddings.get_text(i)
        text = text[:40] + "..." if text and len(text) > 40 else text or "-"

        table.add_row(
            str(i),
            embeddings.get_id(i) or "-",
            vec_str,
            text,
        )

    console.print(table)


if __name__ == "__main__":
    main()
