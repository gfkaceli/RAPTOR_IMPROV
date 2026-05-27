"""
Clustering subpackage for the RAPTOR improvements thesis.

Public API:
    BaseClusterer       -- abstract base class for all clusterers
    ClusteringResult    -- rich result dataclass
    GMMClusterer        -- upstream RAPTOR baseline (GMM + UMAP)
    LeidenClusterer     -- multi-feature graph Leiden (thesis contribution 1)
    LeidenConfig        -- all Leiden hyperparameters in one place

To add a new clusterer (e.g. K-Means, DBSCAN, Agglomerative):
    1. Create raptor/clustering/<name>.py
    2. Subclass BaseClusterer and implement `_cluster_embeddings`.
    3. Export the class here in __all__.
    4. The ablation runner picks it up automatically.

To use a clusterer with the upstream RAPTOR builder:
    >>> from raptor.clustering import LeidenClusterer, LeidenConfig
    >>> from raptor.cluster_tree_builder import ClusterTreeConfig
    >>> from raptor.RetrievalAugmentation import RetrievalAugmentationConfig
    >>>
    >>> clusterer = LeidenClusterer(
    ...     config=LeidenConfig(resolution=1.0, k_neighbors=15),
    ...     verbose=True,
    ... )
    >>> tree_config = ClusterTreeConfig(
    ...     clustering_algorithm=clusterer,   # pass the INSTANCE
    ...     clustering_params={},             # all params live on the instance now
    ... )
    >>> ra_config = RetrievalAugmentationConfig(tree_builder_config=tree_config)

Note that the upstream `ClusterTreeBuilder.construct_tree` calls
`self.clustering_algorithm.perform_clustering(...)`. Passing an instance works
because Python binds `self` automatically; passing a class (like the upstream
default `RAPTOR_Clustering`) works because `perform_clustering` was defined
without an explicit `self` parameter as a quasi-classmethod. Our BaseClusterer
uses `self`, so you MUST pass an instance.
"""

from .base import BaseClusterer, ClusteringResult
from .gmm import GMMClusterer
from .leiden import LeidenClusterer, LeidenConfig
from .kmeans import KMeansClusterer

__all__ = [
    "BaseClusterer",
    "ClusteringResult",
    "GMMClusterer",
    "LeidenClusterer",
    "LeidenConfig",
    "KMeansClusterer"
]

