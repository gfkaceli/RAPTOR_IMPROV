"""
AgglomerativeClusterer: Hierarchical agglomerative clustering for RAPTOR.

Laitenberger (Stanford CS224N, 2024) replaced GMMs with agglomerative
clustering using average linkage and cosine distance, producing deeper trees
and consistently outperforming GMMs on QASPER. His dendrogram cuts at n/3
and n/6 were heuristic and likely dataset-specific.

This implementation improves on Laitenberger by offering three cut strategies:
    - "silhouette": try multiple cut thresholds, pick the one with best
      silhouette score. Data-adaptive, no hand-tuned constants.
    - "distance": cut the dendrogram at a percentile of the merge distance
      distribution. Tunable via distance_percentile.
    - "fixed_k": cut to produce exactly K clusters.

Layer-adaptive behavior is supported via k_schedule (same as KMeansClusterer).

Integration:
    1. Drop into raptor/clustering/agglomerative.py
    2. Add to raptor/clustering/__init__.py:
           from .agglomerative import AgglomerativeClusterer
       and add "AgglomerativeClusterer" to __all__
    3. Import and use:
           from raptor.clustering import AgglomerativeClusterer
           clusterer = AgglomerativeClusterer(random_state=224)

Dependencies:
    scikit-learn, scipy (both already in RAPTOR's requirements.txt)

References:
    - Laitenberger (2024), Expanding Horizons in RAG. Stanford CS224N.
    - Müllner (2011), Modern hierarchical, agglomerative clustering algorithms.
      arXiv:1109.2378.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from .base import BaseClusterer

logger = logging.getLogger(__name__)


class AgglomerativeClusterer(BaseClusterer):
    """
    Agglomerative clustering with cosine distance and data-adaptive cutting.

    Parameters
    ----------
    linkage : str
        Linkage criterion: "average" (default, used by Laitenberger), "complete",
        "single", or "ward". Ward requires euclidean metric and is not
        recommended for normalized embeddings.
    cut_strategy : str
        "silhouette" — try K=min_k..max_k, pick best silhouette (default).
        "distance" — cut at a percentile of the merge distance distribution.
        "fixed_k" — cut to produce exactly fixed_k clusters.
    min_k : int
        Minimum number of clusters. Default 3.
    max_k : int
        Maximum number of clusters for silhouette search. Default 10.
    fixed_k : int
        Number of clusters when cut_strategy="fixed_k". Default 5.
    distance_percentile : float
        Percentile (0-100) of merge distances at which to cut when
        cut_strategy="distance". Higher = fewer clusters. Default 70.
    k_schedule : dict, optional
        Per-layer multiplier on K (same as KMeansClusterer).
    **base_kwargs
        Forwarded to BaseClusterer.

    Example
    -------
    >>> from raptor.clustering import AgglomerativeClusterer
    >>> clusterer = AgglomerativeClusterer(cut_strategy="silhouette", random_state=224)
    >>> ClusterTreeConfig(clustering_algorithm=clusterer, ...)
    """

    algorithm_name = "agglomerative"
    supports_soft_clustering = False

    def __init__(
        self,
        *,
        linkage: str = "average",
        cut_strategy: str = "silhouette",
        min_k: int = 3,
        max_k: int = 10,
        fixed_k: int = 5,
        distance_percentile: float = 70.0,
        k_schedule: Optional[Dict[int, float]] = None,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)
        if linkage not in ("average", "complete", "single", "ward"):
            raise ValueError(f"linkage must be average/complete/single/ward, got '{linkage}'")
        if cut_strategy not in ("silhouette", "distance", "fixed_k"):
            raise ValueError(f"cut_strategy must be silhouette/distance/fixed_k, got '{cut_strategy}'")
        self.linkage = linkage
        self.cut_strategy = cut_strategy
        self.min_k = min_k
        self.max_k = max_k
        self.fixed_k = fixed_k
        self.distance_percentile = distance_percentile
        self.k_schedule = k_schedule

        self._last_k = 0
        self._last_silhouette = float("nan")
        self._last_cophenetic_corr = float("nan")

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        *,
        layer: Optional[int] = None,
    ) -> np.ndarray:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        n = embeddings.shape[0]
        if n <= 1:
            return np.zeros(n, dtype=int)
        if n == 2:
            return np.array([0, 1], dtype=int)

        # Compute distance matrix and linkage
        metric = "euclidean" if self.linkage == "ward" else "cosine"
        dists = pdist(embeddings, metric=metric)
        # Guard against NaN from zero-norm embeddings
        dists = np.nan_to_num(dists, nan=1.0)
        Z = linkage(dists, method=self.linkage)

        # Compute cophenetic correlation for quality logging
        try:
            from scipy.cluster.hierarchy import cophenet
            c, _ = cophenet(Z, dists)
            self._last_cophenetic_corr = float(c)
        except Exception:
            self._last_cophenetic_corr = float("nan")

        # Determine K
        k = self._pick_k(embeddings, Z, dists, n)

        # Apply layer schedule
        if self.k_schedule is not None and layer is not None and layer in self.k_schedule:
            k = max(2, int(k * self.k_schedule[layer]))

        k = max(2, min(k, n))
        labels = fcluster(Z, t=k, criterion="maxclust") - 1  # fcluster is 1-indexed

        self._last_k = k

        # Silhouette for logging
        if k >= 2 and k < n:
            try:
                from sklearn.metrics import silhouette_score
                self._last_silhouette = float(
                    silhouette_score(embeddings, labels, metric="cosine")
                )
            except Exception:
                self._last_silhouette = float("nan")

        return labels

    def _pick_k(
        self,
        embeddings: np.ndarray,
        Z: np.ndarray,
        dists: np.ndarray,
        n: int,
    ) -> int:
        """Choose K based on the configured cut strategy."""
        if self.cut_strategy == "fixed_k":
            return max(self.min_k, self.fixed_k)

        if self.cut_strategy == "distance":
            return self._distance_cut(Z, dists, n)

        return self._silhouette_cut(embeddings, Z, n)

    def _distance_cut(self, Z: np.ndarray, dists: np.ndarray, n: int) -> int:
        """
        Cut the dendrogram at a percentile of merge distances.
        The merge distances are in Z[:, 2]. Higher percentile = cut higher
        in the tree = fewer clusters.
        """
        from scipy.cluster.hierarchy import fcluster

        merge_dists = Z[:, 2]
        threshold = np.percentile(merge_dists, self.distance_percentile)
        labels = fcluster(Z, t=threshold, criterion="distance")
        k = len(set(labels))
        return max(self.min_k, k)

    def _silhouette_cut(self, embeddings: np.ndarray, Z: np.ndarray, n: int) -> int:
        """Try K=min_k..max_k via fcluster(maxclust), pick best silhouette."""
        from scipy.cluster.hierarchy import fcluster
        from sklearn.metrics import silhouette_score

        lo = max(2, self.min_k)
        hi = min(self.max_k, n - 1)
        if hi < lo:
            return lo

        best_k = lo
        best_score = -1.0

        for k in range(lo, hi + 1):
            labels = fcluster(Z, t=k, criterion="maxclust") - 1
            if len(set(labels)) < 2:
                continue
            try:
                score = silhouette_score(embeddings, labels, metric="cosine")
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_k = k

        if self.verbose:
            logger.info(
                "Agglomerative silhouette: best K=%d (score=%.4f) from [%d, %d]",
                best_k, best_score, lo, hi,
            )

        return best_k

    def _compute_metrics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        base = super()._compute_metrics(embeddings, labels)
        base["k"] = float(self._last_k)
        base["silhouette"] = self._last_silhouette
        base["cophenetic_corr"] = self._last_cophenetic_corr
        return base
