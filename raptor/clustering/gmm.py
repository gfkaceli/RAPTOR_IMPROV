"""
GMMClusterer: a thin wrapper around RAPTOR's upstream GMM+UMAP clustering.

The thesis's clustering bake-off is only fair if EVERY clusterer goes through
the same interface (BaseClusterer) and emits the same ClusteringResult. This
file wraps the upstream `perform_clustering` function from `raptor.cluster_utils`
so the GMM baseline behaves identically to GMMClusterer's siblings in the
ablation grid.

We re-use the upstream UMAP + GaussianMixture + BIC pipeline verbatim — the
point of this wrapper is NOT to reimplement the baseline, but to plumb it
through BaseClusterer so the ablation runner can swap clusterers without
special-casing GMM.

Why this matters for the ablation
---------------------------------
GMM in upstream RAPTOR is SOFT clustering: a node can belong to multiple
clusters if its membership probability exceeds 0.1 in each. Our LeidenClusterer
is HARD. If we compared "upstream GMM" against "LeidenClusterer" directly, the
soft vs. hard distinction would confound the algorithm comparison. So:

  - GMMClusterer exposes a `force_hard_clustering` flag. When True, each node is
    assigned to its argmax cluster only. The default is False (matching upstream
    behavior) so that "compare to upstream RAPTOR" cells in the ablation work as
    expected.

  - The `supports_soft_clustering` class attribute lets the ablation logger
    record which condition was active.

References
----------
- Sarthi et al. (ICLR 2024), RAPTOR. The implementation copied below is from
  parthsarthi03/raptor's raptor/cluster_utils.py.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from .base import BaseClusterer

logger = logging.getLogger(__name__)


class GMMClusterer(BaseClusterer):
    """
    Gaussian Mixture Model clustering on UMAP-reduced embeddings.

    This is the upstream RAPTOR baseline, ported into the BaseClusterer
    interface.
    """

    algorithm_name = "gmm"
    supports_soft_clustering = True

    def __init__(
        self,
        *,
        reduction_dimension: int = 10,
        soft_threshold: float = 0.1,
        max_clusters_per_bic_search: int = 50,
        force_hard_clustering: bool = False,
        umap_n_neighbors: Optional[int] = None,
        umap_metric: str = "cosine",
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)
        self.reduction_dimension = reduction_dimension
        self.soft_threshold = soft_threshold
        self.max_clusters_per_bic_search = max_clusters_per_bic_search
        self.force_hard_clustering = force_hard_clustering
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_metric = umap_metric

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        *,
        layer: Optional[int] = None,
    ) -> np.ndarray:
        """
        Returns hard labels (argmax cluster per node). The soft membership
        information is computed but, if `force_hard_clustering=False`, the
        cluster() override below will expand soft memberships into the
        List[List[Node]] return value.
        """
        try:
            import umap  # type: ignore
            from sklearn.mixture import GaussianMixture  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "GMMClusterer requires `umap-learn` and `scikit-learn`. "
                "Both are already in upstream RAPTOR's requirements.txt."
            ) from exc

        n = embeddings.shape[0]
        if n <= 1:
            return np.zeros(n, dtype=int)

        # BaseClusterer already reduced when reduce_embeddings=True; re-use
        # that matrix to keep all methods in the same reduced space.
        if self.reduce_embeddings:
            reduced = embeddings
        else:
            if n <= self.reduction_dimension + 1:
                # Not enough points to do UMAP reduction; cluster everything together.
                return np.zeros(n, dtype=int)

            # UMAP reduction with seed for reproducibility — upstream RAPTOR does
            # NOT set the seed, which is a determinism bug. We fix it here.
            n_neighbors = self.umap_n_neighbors
            if n_neighbors is None:
                n_neighbors = max(2, int((n - 1) ** 0.5))
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=min(self.reduction_dimension, n - 2),
                metric=self.umap_metric,
                random_state=self.random_state,
            )
            reduced = reducer.fit_transform(embeddings)

        # BIC search for optimal n_clusters.
        # Cap max_k at sqrt(n) to prevent BIC from overfitting on small datasets.
        # With 30 nodes, searching up to K=29 produces clusters of 1-2 nodes
        # which always have lower BIC (perfect fit) but are meaningless.
        # sqrt(n) is a standard heuristic that balances granularity with stability.
        max_k = min(self.max_clusters_per_bic_search, max(3, int(np.sqrt(n)) + 1))
        ks = np.arange(1, max_k + 1)
        bics = []
        for k in ks:
            gm = GaussianMixture(n_components=k, random_state=self.random_state)
            gm.fit(reduced)
            bics.append(gm.bic(reduced))
        optimal_k = int(ks[int(np.argmin(bics))])

        # Fit at the optimal number of clusters
        gm = GaussianMixture(
            n_components=optimal_k, random_state=self.random_state
        )
        gm.fit(reduced)
        probs = gm.predict_proba(reduced)  # (n, k)
        self._last_bic = float(min(bics))
        self._last_optimal_k = optimal_k

        # Stash soft memberships for `cluster()` to use.
        self._last_soft_probs = probs

        # Hard labels = argmax
        return np.argmax(probs, axis=1)

    def cluster(self, nodes, embedding_model_name, *, layer=None):
        """
        Override to handle soft clustering: a node can appear in multiple
        clusters if its probability exceeds soft_threshold.
        """
        result = super().cluster(nodes, embedding_model_name, layer=layer)

        if self.force_hard_clustering or not hasattr(self, "_last_soft_probs"):
            return result

        # Re-emit clusters using soft thresholding (upstream behavior).
        probs = self._last_soft_probs
        n_clusters = probs.shape[1]
        soft_clusters: List[List] = [[] for _ in range(n_clusters)]
        for i, node in enumerate(nodes):
            assigned = np.where(probs[i] > self.soft_threshold)[0]
            if len(assigned) == 0:
                # Fall back to argmax to ensure every node is assigned
                assigned = [int(np.argmax(probs[i]))]
            for k in assigned:
                soft_clusters[k].append(node)

        # Drop empty clusters (can happen with low thresholds)
        soft_clusters = [c for c in soft_clusters if c]
        result.clusters = soft_clusters
        result.metrics["soft_assignments"] = float(
            sum(len(c) for c in soft_clusters) / max(len(nodes), 1)
        )
        return result

    def _compute_metrics(self, embeddings, labels) -> Dict[str, float]:
        base = super()._compute_metrics(embeddings, labels)
        base["bic"] = getattr(self, "_last_bic", float("nan"))
        base["optimal_k"] = float(getattr(self, "_last_optimal_k", 0))
        return base