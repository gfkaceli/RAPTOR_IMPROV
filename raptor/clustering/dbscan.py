"""
DBSCANClusterer: Density-based clustering for RAPTOR hierarchical indexing.

DBSCAN discovers clusters of arbitrary shape without requiring the number of
clusters as input. Unlike GMM, K-Means, and Agglomerative, DBSCAN naturally
identifies noise points — chunks that don't belong to any cluster. This is
useful for RAPTOR because it can separate "orphan" chunks that don't fit
any thematic group.

Noise handling strategies (controlled by noise_strategy):
    - "singleton": each noise point becomes its own 1-node cluster (default).
      No information is lost; the node is promoted to the next layer as-is.
    - "nearest": assign each noise point to the nearest non-noise cluster
      by cosine similarity to centroids. Produces cleaner clusters but may
      force unrelated chunks together.
    - "drop": noise points are excluded from clustering and promoted directly
      to the parent level without summarization. (Not yet implemented —
      requires changes to ClusterTreeBuilder.)

The key hyperparameter is eps (neighborhood radius). We offer auto-tuning
via the k-distance graph method: compute the distance to the k-th nearest
neighbor for all points, sort, and pick eps at the "knee" of the curve.

Integration:
    Same as KMeansClusterer and AgglomerativeClusterer — drop into
    raptor/clustering/dbscan.py, add to __init__.py, import, and use.

Dependencies:
    scikit-learn (already in requirements.txt)

References:
    - Ester et al. (1996), A Density-Based Algorithm for Discovering Clusters
      in Large Spatial Databases with Noise. KDD.
    - Schubert et al. (2017), DBSCAN Revisited. ACM TODS.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from .base import BaseClusterer

logger = logging.getLogger(__name__)


class DBSCANClusterer(BaseClusterer):
    """
    DBSCAN clustering with automatic eps tuning and noise handling.

    Parameters
    ----------
    eps : float, optional
        Maximum neighborhood distance. If None (default), eps is estimated
        automatically using the k-distance knee method.
    min_samples : int
        Minimum points to form a dense region. Default 2 (appropriate for
        RAPTOR where clusters can be small).
    metric : str
        Distance metric. Default "cosine".
    noise_strategy : str
        How to handle noise points: "singleton" (default) or "nearest".
    eps_percentile : float
        Percentile of the k-distance curve to use as eps when auto-tuning.
        Default 80. Higher = larger eps = fewer but bigger clusters.
    k_neighbors_for_eps : int
        The k in k-distance graph for auto eps. Default 4.
    eps_schedule : dict, optional
        Per-layer eps multiplier. E.g. {0: 1.0, 1: 1.5} — larger eps at
        higher layers produces fewer, broader clusters.
    **base_kwargs
        Forwarded to BaseClusterer.

    Example
    -------
    >>> from raptor.clustering import DBSCANClusterer
    >>> clusterer = DBSCANClusterer(noise_strategy="singleton", random_state=224)
    >>> ClusterTreeConfig(clustering_algorithm=clusterer, ...)
    """

    algorithm_name = "dbscan"
    supports_soft_clustering = False

    def __init__(
        self,
        *,
        eps: Optional[float] = None,
        min_samples: int = 2,
        metric: str = "cosine",
        noise_strategy: str = "singleton",
        eps_percentile: float = 80.0,
        k_neighbors_for_eps: int = 4,
        eps_schedule: Optional[Dict[int, float]] = None,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)
        if noise_strategy not in ("singleton", "nearest"):
            raise ValueError(
                f"noise_strategy must be 'singleton' or 'nearest', got '{noise_strategy}'"
            )
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.noise_strategy = noise_strategy
        self.eps_percentile = eps_percentile
        self.k_neighbors_for_eps = k_neighbors_for_eps
        self.eps_schedule = eps_schedule

        self._last_eps = 0.0
        self._last_n_noise = 0
        self._last_silhouette = float("nan")

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        *,
        layer: Optional[int] = None,
    ) -> np.ndarray:
        from sklearn.cluster import DBSCAN

        n = embeddings.shape[0]
        if n <= 1:
            return np.zeros(n, dtype=int)

        # Determine eps
        eps = self.eps if self.eps is not None else self._auto_eps(embeddings)

        # Apply layer schedule
        if self.eps_schedule is not None and layer is not None and layer in self.eps_schedule:
            eps = eps * self.eps_schedule[layer]

        self._last_eps = eps

        db = DBSCAN(
            eps=eps,
            min_samples=self.min_samples,
            metric=self.metric,
        )
        labels = db.fit_predict(embeddings)

        self._last_n_noise = int((labels == -1).sum())

        # Handle noise based on strategy
        if self._last_n_noise > 0 and self.noise_strategy == "nearest":
            labels = self._assign_noise_to_nearest(embeddings, labels)

        # If DBSCAN produced only noise (all -1) or a single cluster,
        # fall back to putting everything in one cluster. This can happen
        # when eps is too small.
        unique_labels = set(labels)
        unique_labels.discard(-1)
        if len(unique_labels) == 0:
            if self.verbose:
                logger.warning(
                    "DBSCAN: all points are noise (eps=%.4f). Falling back to "
                    "single cluster.", eps,
                )
            return np.zeros(n, dtype=int)

        # Silhouette for logging (only on non-noise points if singletons exist)
        non_noise_mask = labels != -1
        if non_noise_mask.sum() >= 2 and len(set(labels[non_noise_mask])) >= 2:
            try:
                from sklearn.metrics import silhouette_score
                self._last_silhouette = float(
                    silhouette_score(
                        embeddings[non_noise_mask],
                        labels[non_noise_mask],
                        metric="cosine",
                    )
                )
            except Exception:
                self._last_silhouette = float("nan")

        return labels

    def _auto_eps(self, embeddings: np.ndarray) -> float:
        """
        Estimate eps using the k-distance graph method.

        For each point, compute the distance to its k-th nearest neighbor.
        Sort these distances. The "knee" of the sorted curve is a good eps
        estimate. We approximate the knee by taking a percentile of the
        k-distances.
        """
        from sklearn.neighbors import NearestNeighbors

        n = embeddings.shape[0]
        k = min(self.k_neighbors_for_eps, n - 1)
        if k < 1:
            return 0.5  # fallback for very small inputs

        nn = NearestNeighbors(n_neighbors=k, metric=self.metric)
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        # k-th neighbor distance (last column)
        k_distances = distances[:, -1]
        k_distances.sort()

        eps = float(np.percentile(k_distances, self.eps_percentile))

        # Sanity bounds: eps shouldn't be 0 (identical points) or > 2 (cosine max)
        if self.metric == "cosine":
            eps = max(0.01, min(eps, 1.5))
        else:
            eps = max(1e-6, eps)

        if self.verbose:
            logger.info(
                "DBSCAN auto-eps: %.4f (percentile=%.0f of %d-distance graph)",
                eps, self.eps_percentile, k,
            )

        return eps

    def _assign_noise_to_nearest(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Assign noise points (-1) to the nearest non-noise cluster centroid."""
        labels = labels.copy()
        noise_idx = np.where(labels == -1)[0]
        non_noise_labels = set(labels)
        non_noise_labels.discard(-1)

        if not non_noise_labels:
            return labels

        # Compute centroids of non-noise clusters
        cluster_ids = sorted(non_noise_labels)
        centroids = np.vstack([
            embeddings[labels == cid].mean(axis=0) for cid in cluster_ids
        ])
        # Normalize for cosine similarity
        centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

        for idx in noise_idx:
            point = embeddings[idx]
            point_norm = point / (np.linalg.norm(point) + 1e-12)
            sims = centroids_norm @ point_norm
            best_cluster = cluster_ids[int(np.argmax(sims))]
            labels[idx] = best_cluster

        return labels

    def _compute_metrics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        base = super()._compute_metrics(embeddings, labels)
        base["eps"] = self._last_eps
        base["n_noise_before_handling"] = float(self._last_n_noise)
        base["silhouette"] = self._last_silhouette
        return base
