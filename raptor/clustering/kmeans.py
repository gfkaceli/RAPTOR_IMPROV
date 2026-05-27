"""
KMeansClusterer: K-Means clustering for RAPTOR hierarchical indexing.

Zhou et al. (2025, "Triplet-Driven Thinking RAG") showed that simply replacing
GMM with K-Means in RAPTOR produces comparable quality with better computational
efficiency. This implementation adds automatic K selection via silhouette score
and layer-adaptive K scaling.

Integration:
    1. Drop this file into raptor/clustering/kmeans.py
    2. Add to raptor/clustering/__init__.py:
           from .kmeans import KMeansClusterer
       and add "KMeansClusterer" to __all__
    3. Import and use like GMMClusterer:
           from raptor.clustering import KMeansClusterer
           clusterer = KMeansClusterer(random_state=224)

Dependencies:
    scikit-learn (already in RAPTOR's requirements.txt)

References:
    - Zhou et al. (2025), Triplet-Driven Thinking RAG. arXiv preprint.
    - Laitenberger (2024), Expanding Horizons in RAG. Stanford CS224N.
    - Arthur & Vassilvitskii (2007), k-means++: The Advantages of Careful
      Seeding. SODA.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseClusterer

logger = logging.getLogger(__name__)


class KMeansClusterer(BaseClusterer):
    """
    K-Means clustering with automatic K selection.

    K is chosen by one of three strategies:
        - "silhouette": try K=2..max_k, pick the K with the highest silhouette
          score. Most principled but slower.
        - "sqrt": K = ceil(sqrt(n_nodes)). Simple, fast, works well in practice.
        - "fixed": use a fixed K passed at construction time.

    Layer-adaptive K scaling is supported via k_schedule: a dict mapping
    layer index to a multiplier on the base K. For example, {0: 1.0, 1: 0.5}
    halves the number of clusters at layer 1, producing broader thematic groups
    higher in the tree.

    Parameters
    ----------
    k_strategy : str
        "silhouette", "sqrt", or "fixed". Default "silhouette".
    fixed_k : int
        Number of clusters when k_strategy="fixed". Ignored otherwise.
    max_k : int
        Maximum K to try during silhouette search. Default 10.
    k_schedule : dict, optional
        Per-layer multiplier on K. Keys are layer indices, values are floats.
        E.g. {0: 1.0, 1: 0.6, 2: 0.3} — fewer clusters at higher layers.
    n_init : int
        Number of K-Means initializations. Default 10 (sklearn default).
    **base_kwargs
        Forwarded to BaseClusterer (random_state, max_length_in_cluster, etc.)

    Example
    -------
    >>> from raptor.clustering import KMeansClusterer
    >>> clusterer = KMeansClusterer(k_strategy="silhouette", random_state=224)
    >>> # Pass to ClusterTreeConfig:
    >>> ClusterTreeConfig(clustering_algorithm=clusterer, ...)
    """

    algorithm_name = "kmeans"
    supports_soft_clustering = False

    def __init__(
        self,
        *,
        k_strategy: str = "silhouette",
        fixed_k: int = 5,
        min_k: int = 3,
        max_k: int = 10,
        k_schedule: Optional[Dict[int, float]] = None,
        n_init: int = 10,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)
        if k_strategy not in ("silhouette", "sqrt", "fixed"):
            raise ValueError(
                f"k_strategy must be 'silhouette', 'sqrt', or 'fixed', got '{k_strategy}'"
            )
        self.k_strategy = k_strategy
        self.fixed_k = fixed_k
        self.min_k = min_k
        self.max_k = max_k
        self.k_schedule = k_schedule
        self.n_init = n_init

        # Stashed per-call for metrics
        self._last_k = 0
        self._last_inertia = 0.0
        self._last_silhouette = float("nan")

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        *,
        layer: Optional[int] = None,
    ) -> np.ndarray:
        from sklearn.cluster import KMeans

        n = embeddings.shape[0]
        if n <= 1:
            return np.zeros(n, dtype=int)

        # Determine K
        base_k = self._pick_k(embeddings)

        # Apply layer schedule
        if self.k_schedule is not None and layer is not None and layer in self.k_schedule:
            base_k = max(2, int(base_k * self.k_schedule[layer]))

        k = min(base_k, n)
        if k < 2:
            return np.zeros(n, dtype=int)

        km = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        labels = km.fit_predict(embeddings)

        self._last_k = k
        self._last_inertia = float(km.inertia_)

        # Compute silhouette for logging (cheap at this scale)
        if k >= 2 and k < n:
            try:
                from sklearn.metrics import silhouette_score
                self._last_silhouette = float(
                    silhouette_score(embeddings, labels, metric="cosine")
                )
            except Exception:
                self._last_silhouette = float("nan")

        return labels

    def _pick_k(self, embeddings: np.ndarray) -> int:
        """Choose K based on the configured strategy, enforcing min_k."""
        n = embeddings.shape[0]

        if self.k_strategy == "fixed":
            return max(self.min_k, self.fixed_k)

        if self.k_strategy == "sqrt":
            return max(self.min_k, int(np.ceil(np.sqrt(n))))

        # silhouette search
        return self._silhouette_search(embeddings)

    def _silhouette_search(self, embeddings: np.ndarray) -> int:
        """Try K=min_k..max_k, return the K with highest silhouette score."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        n = embeddings.shape[0]
        lo = max(2, self.min_k)
        hi = min(self.max_k, n - 1)
        if hi < lo:
            return lo

        best_k = lo
        best_score = -1.0

        for k in range(lo, hi + 1):
            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
            )
            labels = km.fit_predict(embeddings)
            # silhouette_score needs at least 2 distinct labels
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
                "KMeans silhouette search: best K=%d (score=%.4f) from range [%d, %d]",
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
        base["inertia"] = self._last_inertia
        base["silhouette"] = self._last_silhouette
        return base