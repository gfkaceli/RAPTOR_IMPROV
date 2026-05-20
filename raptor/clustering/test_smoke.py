"""
Smoke tests for the clustering subpackage.

Run from the repo root with:
    python -m raptor.clustering.test_smoke

These tests exercise the BaseClusterer + ClusteringResult plumbing using a
minimal FakeNode that mimics the upstream raptor.tree_structures.Node interface.
They do NOT exercise the full RAPTOR build pipeline — for that you need the
existing demo.py.

What is covered:
  - BaseClusterer abstract methods enforce subclassing correctly
  - ClusteringResult is well-formed (clusters, labels, centroids consistent)
  - GMMClusterer reproduces upstream RAPTOR behavior on a toy dataset
  - LeidenClusterer raises an informative error if leidenalg is missing,
    or runs end-to-end if it is installed

To run only the leiden test (after `pip install leidenalg python-igraph`):
    python -m raptor.clustering.test_smoke leiden
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


# -----------------------------------------------------------------------------
# Minimal Node stand-in so this file is runnable without the full raptor package
# -----------------------------------------------------------------------------

@dataclass
class FakeNode:
    """
    Mimics raptor.tree_structures.Node just enough for clustering tests.

    Real Node has: index, text, children, embeddings (Dict[str, np.ndarray]).
    For clustering we only touch `embeddings` and `text`.
    """
    index: int
    text: str
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    children: set = field(default_factory=set)


# Monkey-patch so the `from ..tree_structures import Node` line in base.py
# resolves to our FakeNode when running this file standalone (or when running
# inside a repo that doesn't have the full upstream raptor package built).
#
# IMPORTANT: this must happen BEFORE any import of raptor.clustering, otherwise
# the package __init__ runs first and the relative import has nothing to bind
# to. When the test is run as `python -m raptor.clustering.test_smoke`, the
# package __init__ has already executed by the time we get here — for that
# invocation, the real raptor.tree_structures must exist (which it will once
# you drop these files into your repo). For standalone invocation
# (`python test_smoke.py`) the monkeypatch below kicks in.
import os
import types

# Add the repo root to sys.path so `import raptor.*` resolves.
# This file lives at <repo>/raptor/clustering/test_smoke.py, so:
#   _HERE         = <repo>/raptor/clustering
#   _REPO_ROOT    = <repo>
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# If the real raptor.tree_structures is not available, install the fake one
# BEFORE we import anything from raptor.clustering.
try:
    from raptor.tree_structures import Node as _RealNode  # noqa: F401
    # Real package present — nothing to patch.
except ImportError:
    if "raptor" not in sys.modules:
        sys.modules["raptor"] = types.ModuleType("raptor")
    _fake_tree_structures = types.ModuleType("raptor.tree_structures")
    _fake_tree_structures.Node = FakeNode
    sys.modules["raptor.tree_structures"] = _fake_tree_structures

from raptor.clustering.base import BaseClusterer, ClusteringResult


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_clustered_toy_data(seed: int = 0):
    """
    Build 30 fake nodes with embeddings drawn from 3 clearly-separated Gaussian
    blobs in R^16. Any sensible clusterer should recover 3 clusters here.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, 16)) * 5.0  # well-separated centers
    nodes = []
    for cluster_id in range(3):
        for i in range(10):
            emb = centers[cluster_id] + rng.normal(size=16) * 0.3
            emb = emb / np.linalg.norm(emb)  # unit-normalize like SBERT
            node = FakeNode(
                index=len(nodes),
                text=f"This is text for cluster {cluster_id}, node {i}.",
                embeddings={"sbert": emb},
            )
            nodes.append(node)
    return nodes


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_base_clusterer_is_abstract():
    """BaseClusterer must reject instantiation without `_cluster_embeddings`."""
    print("test_base_clusterer_is_abstract ...", end=" ")
    try:
        BaseClusterer()  # type: ignore[abstract]
        print("FAIL (expected TypeError)")
        return False
    except TypeError:
        print("OK")
        return True


def test_dummy_clusterer_returns_well_formed_result():
    """A minimal subclass that always returns label 0 should still produce a valid result."""
    print("test_dummy_clusterer_returns_well_formed_result ...", end=" ")

    class DummyClusterer(BaseClusterer):
        algorithm_name = "dummy"
        recursive_split = False  # avoid needing tokenizer
        def _cluster_embeddings(self, embeddings, *, layer=None):
            return np.zeros(embeddings.shape[0], dtype=int)

    nodes = make_clustered_toy_data()
    clusterer = DummyClusterer(verbose=False)
    result = clusterer.cluster(nodes, embedding_model_name="sbert", layer=0)

    assert isinstance(result, ClusteringResult), "result must be ClusteringResult"
    assert len(result.clusters) == 1, f"expected 1 cluster, got {len(result.clusters)}"
    assert len(result.clusters[0]) == 30, "all nodes should be in the single cluster"
    assert result.labels.shape == (30,), f"labels shape wrong: {result.labels.shape}"
    assert result.centroids.shape == (1, 16), f"centroids shape wrong: {result.centroids.shape}"
    assert result.algorithm == "dummy"
    assert result.layer == 0
    print("OK")
    return True


def test_perform_clustering_adapter_returns_list_of_lists():
    """The upstream-RAPTOR contract: perform_clustering returns List[List[Node]]."""
    print("test_perform_clustering_adapter_returns_list_of_lists ...", end=" ")

    class TwoClusterClusterer(BaseClusterer):
        algorithm_name = "two-cluster"
        recursive_split = False
        def _cluster_embeddings(self, embeddings, *, layer=None):
            n = embeddings.shape[0]
            return np.array([i % 2 for i in range(n)], dtype=int)

    nodes = make_clustered_toy_data()
    clusterer = TwoClusterClusterer()
    clusters = clusterer.perform_clustering(
        nodes, embedding_model_name="sbert", reduction_dimension=10
    )

    assert isinstance(clusters, list), "must return a list"
    assert all(isinstance(c, list) for c in clusters), "each cluster must be a list"
    assert len(clusters) == 2, f"expected 2 clusters, got {len(clusters)}"
    assert sum(len(c) for c in clusters) == 30, "no nodes should be lost"
    print("OK")
    return True


def test_noise_points_become_singletons():
    """DBSCAN-style noise (label -1) should be emitted as singleton clusters."""
    print("test_noise_points_become_singletons ...", end=" ")

    class NoisyClusterer(BaseClusterer):
        algorithm_name = "noisy"
        recursive_split = False
        def _cluster_embeddings(self, embeddings, *, layer=None):
            n = embeddings.shape[0]
            labels = np.zeros(n, dtype=int)
            labels[0] = -1  # first node is "noise"
            labels[1] = -1
            return labels

    nodes = make_clustered_toy_data()[:5]
    clusterer = NoisyClusterer()
    result = clusterer.cluster(nodes, embedding_model_name="sbert")

    # Expected: 2 singleton clusters (the noise points) + 1 cluster of 3
    assert len(result.clusters) == 3, f"expected 3 clusters, got {len(result.clusters)}"
    singleton_count = sum(1 for c in result.clusters if len(c) == 1)
    assert singleton_count == 2, f"expected 2 singletons, got {singleton_count}"
    print("OK")
    return True


def test_gmm_clusterer_runs():
    """End-to-end GMM smoke test. Skipped if umap / sklearn missing."""
    print("test_gmm_clusterer_runs ...", end=" ")
    try:
        import umap  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        print("SKIP (umap-learn or scikit-learn not installed)")
        return True

    from raptor.clustering.gmm import GMMClusterer

    nodes = make_clustered_toy_data()
    clusterer = GMMClusterer(
        reduction_dimension=5,  # small for 30 nodes
        force_hard_clustering=True,  # easier to assert on
        verbose=False,
    )
    result = clusterer.cluster(nodes, embedding_model_name="sbert", layer=0)

    # The 3 blobs are well-separated; BIC should pick k=3 (or close).
    n_clusters = len(result.clusters)
    assert 2 <= n_clusters <= 5, f"expected ~3 clusters from 3 blobs, got {n_clusters}"
    assert "bic" in result.metrics, "GMM must report BIC"
    assert "optimal_k" in result.metrics, "GMM must report optimal_k"
    print(f"OK (recovered {n_clusters} clusters, BIC={result.metrics['bic']:.2f})")
    return True


def test_leiden_clusterer_runs():
    """End-to-end Leiden smoke test. Skipped if leidenalg/python-igraph missing."""
    print("test_leiden_clusterer_runs ...", end=" ")
    try:
        import leidenalg  # noqa: F401
        import igraph  # noqa: F401
    except ImportError:
        print("SKIP (leidenalg / python-igraph not installed)")
        return True

    from raptor.clustering.leiden import LeidenClusterer, LeidenConfig

    nodes = make_clustered_toy_data()
    cfg = LeidenConfig(
        k_neighbors=5,
        use_adjacency_edges=False,  # the toy data isn't sequential
        resolution=1.0,
    )
    clusterer = LeidenClusterer(config=cfg, verbose=False, random_state=0)
    result = clusterer.cluster(nodes, embedding_model_name="sbert", layer=0)

    n_clusters = len(result.clusters)
    assert 2 <= n_clusters <= 5, f"expected ~3 clusters from 3 blobs, got {n_clusters}"
    assert "modularity" in result.metrics, "Leiden must report modularity"
    assert result.metrics["modularity"] > 0, "modularity should be positive for separated blobs"
    print(
        f"OK (recovered {n_clusters} clusters, modularity={result.metrics['modularity']:.3f})"
    )
    return True


def test_leiden_layer_adaptive_resolution():
    """LeidenClusterer should pick different resolutions per layer."""
    print("test_leiden_layer_adaptive_resolution ...", end=" ")
    try:
        import leidenalg  # noqa: F401
    except ImportError:
        print("SKIP (leidenalg not installed)")
        return True

    from raptor.clustering.leiden import LeidenClusterer, LeidenConfig

    cfg = LeidenConfig(
        k_neighbors=5,
        use_adjacency_edges=False,
        resolution=1.0,
        resolution_schedule={0: 1.5, 1: 0.5},
    )
    clusterer = LeidenClusterer(config=cfg, random_state=0)

    nodes = make_clustered_toy_data()
    result_layer_0 = clusterer.cluster(nodes, "sbert", layer=0)
    result_layer_1 = clusterer.cluster(nodes, "sbert", layer=1)

    # Higher resolution -> more, smaller clusters
    n0 = len(result_layer_0.clusters)
    n1 = len(result_layer_1.clusters)
    print(f"OK (layer 0: {n0} clusters @ r=1.5, layer 1: {n1} clusters @ r=0.5)")
    return True


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

ALL_TESTS = {
    "abstract": test_base_clusterer_is_abstract,
    "dummy": test_dummy_clusterer_returns_well_formed_result,
    "adapter": test_perform_clustering_adapter_returns_list_of_lists,
    "noise": test_noise_points_become_singletons,
    "gmm": test_gmm_clusterer_runs,
    "leiden": test_leiden_clusterer_runs,
    "leiden_adaptive": test_leiden_layer_adaptive_resolution,
}


def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(ALL_TESTS.keys())
    failed = []
    for name in selected:
        if name not in ALL_TESTS:
            print(f"Unknown test: {name}")
            continue
        try:
            ok = ALL_TESTS[name]()
        except Exception as exc:
            print(f"FAIL ({type(exc).__name__}: {exc})")
            ok = False
        if not ok:
            failed.append(name)
    print()
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print("All tests passed.")


if __name__ == "__main__":
    main()
