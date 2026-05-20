"""
BaseClusterer: the abstraction every clustering algorithm in the thesis ablation
plugs into.

Design contract
---------------
1. Compatible with upstream RAPTOR's ClusterTreeBuilder. That builder calls

       clusters = self.clustering_algorithm.perform_clustering(
           node_list_current_layer,
           self.cluster_embedding_model,
           reduction_dimension=self.reduction_dimension,
           **self.clustering_params,
       )

   and expects a return type of List[List[Node]] (a list of clusters, each a
   list of Node objects). We match that signature exactly so that ANY subclass
   can be passed as ClusterTreeConfig(clustering_algorithm=MyClusterer, ...)
   with no changes to cluster_tree_builder.py.

2. Layer-adaptive parameters are passed via the `layer` kwarg. Upstream RAPTOR
   does NOT currently pass `layer`; you patch cluster_tree_builder.py with a
   single-line change (see docstring of `LayerAdaptiveMixin` below). Until that
   patch lands, subclasses gracefully handle layer=None by falling back to a
   default resolution.

3. Soft vs hard clustering is explicit. The base class exposes a
   `supports_soft_clustering` class attribute that the ablation runner reads to
   log whether the clusterer assigns nodes to multiple clusters. This matters
   because GMM (soft) and Leiden (hard) are not strictly comparable on metrics
   that count cluster overlap.

4. Determinism. Every clusterer takes a `random_state` and is expected to be
   bitwise reproducible given the same seed. UMAP, in particular, is NOT
   deterministic by default — subclasses must pass `random_state` into UMAP.

References
----------
- Sarthi et al. (ICLR 2024), RAPTOR: Recursive Abstractive Processing for
  Tree-Organized Retrieval. arXiv:2401.18059. The original GMM+UMAP pipeline.
- Traag, Waltman & van Eck (2019), "From Louvain to Leiden: Guaranteeing
  Well-Connected Communities," Scientific Reports 9:5233. The Leiden
  algorithm used by LeidenClusterer.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# We re-export the upstream Node type so subclasses can type-annotate against it.
# This is the ONLY upstream import in this file — all clusterers should depend on
# this module rather than reaching into raptor.cluster_utils directly.
from ..tree_structures import Node


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------

@dataclass
class ClusteringResult:
    """
    Rich result object that carries more than just the cluster assignment.

    The upstream RAPTOR contract only requires List[List[Node]], but for the
    ablation we want to log additional structural metrics WITHOUT re-running
    clustering. So `BaseClusterer.cluster()` returns this object, and the
    `perform_clustering` adapter unwraps it to List[List[Node]] for the upstream
    builder.
    """
    clusters: List[List[Node]]
    # One row per input node; -1 means "noise" (DBSCAN only). For soft
    # clusterers like GMM the value is the argmax cluster.
    labels: np.ndarray
    # Per-cluster centroid embedding in the ORIGINAL embedding space (not the
    # UMAP-reduced one). Used downstream for centroid-only summarization and
    # for cross-cluster bridge link construction.
    centroids: np.ndarray
    # Free-form structural metrics: e.g. modularity for Leiden, BIC for GMM,
    # silhouette for K-Means. Keys are documented per subclass.
    metrics: Dict[str, float] = field(default_factory=dict)
    # The layer index this clustering was performed at, if known. Useful when
    # the ablation logger picks up a result downstream.
    layer: Optional[int] = None
    # Algorithm name, for logging.
    algorithm: str = "unknown"


# -----------------------------------------------------------------------------
# The base class
# -----------------------------------------------------------------------------

class BaseClusterer(ABC):
    """
    Abstract base for all RAPTOR clusterers in this thesis.

    Subclasses MUST implement `_cluster_embeddings`. They typically do NOT
    override `perform_clustering` — that method exists only as a compatibility
    shim so the upstream `ClusterTreeBuilder` can call us with its existing
    contract.

    Typical subclass implementation:

        class MyClusterer(BaseClusterer):
            algorithm_name = "my-algo"
            supports_soft_clustering = False

            def _cluster_embeddings(self, embeddings, *, layer=None):
                # ... do clustering ...
                return labels_array  # shape (n_nodes,), int

    The base class then takes care of:
      - Extracting embeddings from the Node list
      - Computing centroids in the original embedding space
      - Building the List[List[Node]] return type
      - Wrapping everything in a ClusteringResult
      - Recursive splitting of oversized clusters (mirrors RAPTOR_Clustering)
      - Logging
    """

    # Subclass overrides
    algorithm_name: str = "base"
    supports_soft_clustering: bool = False
    # If True, clusters whose summed token count exceeds max_length_in_cluster
    # are recursively re-clustered. Mirrors upstream RAPTOR behavior. Turn off
    # for the deterministic ablation if you want to compare layer-by-layer
    # behavior without the recursion confound.
    recursive_split: bool = True

    def __init__(
        self,
        *,
        random_state: int = 224,
        max_length_in_cluster: int = 3500,
        tokenizer: Optional[Any] = None,
        verbose: bool = False,
    ) -> None:
        self.random_state = random_state
        self.max_length_in_cluster = max_length_in_cluster
        self.tokenizer = tokenizer  # lazily imported in _token_length
        self.verbose = verbose

    # -------------------------------------------------------------------------
    # Subclass API
    # -------------------------------------------------------------------------

    @abstractmethod
    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        *,
        layer: Optional[int] = None,
    ) -> np.ndarray:
        """
        Cluster the given (n_nodes, embedding_dim) matrix.

        Parameters
        ----------
        embeddings : np.ndarray of shape (n_nodes, embedding_dim)
            Raw embeddings from the nodes. Subclasses may apply UMAP or other
            reduction internally; this method receives the ORIGINAL embeddings.
        layer : int, optional
            Tree layer index this call is for (0 = clustering leaf chunks into
            layer-1 summaries, 1 = clustering layer-1 summaries into layer-2,
            etc.). Layer-adaptive subclasses should use this to vary their
            resolution / number-of-clusters / k-NN parameters.

        Returns
        -------
        labels : np.ndarray of shape (n_nodes,), dtype int
            Cluster label for each node. Use -1 for noise points (DBSCAN).

        Notes
        -----
        For soft-clustering algorithms (GMM), subclasses should override
        `cluster()` directly and provide a labels matrix of shape
        (n_nodes, n_clusters) of probabilities. See `GMMClusterer` for the
        pattern.
        """
        ...

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def cluster(
        self,
        nodes: Sequence[Node],
        embedding_model_name: str,
        *,
        layer: Optional[int] = None,
    ) -> ClusteringResult:
        """
        Cluster a list of Node objects.

        This is the method the ablation logger calls when it wants the rich
        ClusteringResult. The upstream RAPTOR builder calls `perform_clustering`
        below instead, which is a thin adapter that unwraps the result.
        """
        if len(nodes) == 0:
            return ClusteringResult(
                clusters=[],
                labels=np.array([], dtype=int),
                centroids=np.zeros((0, 0)),
                metrics={},
                layer=layer,
                algorithm=self.algorithm_name,
            )

        embeddings = self._extract_embeddings(nodes, embedding_model_name)
        labels = self._cluster_embeddings(embeddings, layer=layer)

        if labels.shape[0] != len(nodes):
            raise RuntimeError(
                f"{self.algorithm_name}: clusterer returned {labels.shape[0]} labels "
                f"for {len(nodes)} nodes. This is a clusterer bug."
            )

        clusters, centroids = self._labels_to_clusters(nodes, embeddings, labels)

        if self.recursive_split:
            clusters = self._maybe_recurse(
                clusters, embedding_model_name, layer=layer
            )

        metrics = self._compute_metrics(embeddings, labels)
        if self.verbose:
            logger.info(
                "%s @ layer=%s: %d input nodes -> %d clusters (metrics=%s)",
                self.algorithm_name, layer, len(nodes), len(clusters), metrics,
            )

        return ClusteringResult(
            clusters=clusters,
            labels=labels,
            centroids=centroids,
            metrics=metrics,
            layer=layer,
            algorithm=self.algorithm_name,
        )

    # -------------------------------------------------------------------------
    # Upstream RAPTOR compatibility shim
    # -------------------------------------------------------------------------
    #
    # RAPTOR's ClusterTreeBuilder calls `self.clustering_algorithm.perform_clustering(...)`
    # — i.e. it expects either a classmethod or a callable on a class. We expose
    # it as a regular method so the builder gets called with `self` bound.
    #
    # To make this drop in without modifying cluster_tree_builder.py, pass an
    # INSTANCE of your clusterer to ClusterTreeConfig:
    #
    #     config = ClusterTreeConfig(
    #         clustering_algorithm=LeidenClusterer(...),  # an INSTANCE
    #     )
    #
    # The instance's `perform_clustering` then has the same calling convention
    # as the upstream `RAPTOR_Clustering.perform_clustering` classmethod.

    def perform_clustering(
        self,
        nodes: List[Node],
        embedding_model_name: str,
        *,
        max_length_in_cluster: Optional[int] = None,
        tokenizer: Optional[Any] = None,
        reduction_dimension: int = 10,  # accepted for API parity, ignored by some clusterers
        threshold: float = 0.1,         # accepted for API parity, used by soft clusterers
        verbose: bool = False,
        layer: Optional[int] = None,
        **kwargs: Any,
    ) -> List[List[Node]]:
        """
        Adapter matching the upstream RAPTOR contract.

        Parameters
        ----------
        nodes : list of Node
        embedding_model_name : str
            Key into Node.embeddings — the name of the embedding model used.
        reduction_dimension : int
            Accepted for parity with upstream. UMAP-based subclasses use it;
            graph-based ones (Leiden) ignore it.
        threshold : float
            Accepted for parity with upstream. Soft clusterers (GMM) use it as
            the probability cutoff for cluster membership.
        layer : int, optional
            Tree layer. The upstream builder does NOT currently pass this — see
            module docstring for the one-line patch to enable it.
        **kwargs
            Extra arguments are accepted and silently ignored, so per-clusterer
            parameter dicts can be passed via `ClusterTreeConfig.clustering_params`
            without breaking other clusterers.
        """
        # Allow per-call overrides (upstream's recursive call passes these)
        if max_length_in_cluster is not None:
            self.max_length_in_cluster = max_length_in_cluster
        if tokenizer is not None:
            self.tokenizer = tokenizer
        if verbose:
            self.verbose = verbose

        # `reduction_dimension` and `threshold` are forwarded to subclasses
        # via instance attributes so they can pick them up if relevant.
        self._reduction_dimension = reduction_dimension
        self._soft_threshold = threshold

        result = self.cluster(nodes, embedding_model_name, layer=layer)
        return result.clusters

    # -------------------------------------------------------------------------
    # Internals — subclasses generally do not override these
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_embeddings(
        nodes: Sequence[Node],
        embedding_model_name: str,
    ) -> np.ndarray:
        """Pull embeddings from the nodes into a single (n, d) array."""
        try:
            return np.array(
                [node.embeddings[embedding_model_name] for node in nodes]
            )
        except KeyError as exc:
            missing = exc.args[0]
            available = sorted(nodes[0].embeddings.keys()) if nodes else []
            raise KeyError(
                f"Embedding key '{missing}' not found on node. "
                f"Available keys on first node: {available}. "
                f"Check that EmbeddingModels and ClusterTreeConfig agree on the model name."
            ) from exc

    @staticmethod
    def _labels_to_clusters(
        nodes: Sequence[Node],
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[List[List[Node]], np.ndarray]:
        """
        Convert per-node labels into clusters + centroids.

        Noise points (label == -1) are emitted as singleton clusters. This
        choice is debatable; the alternative is to drop them or pool them. We
        make them singletons so that no chunk is silently lost from the
        index — losing chunks would break the ablation logging.
        """
        unique = sorted(set(labels.tolist()))
        clusters: List[List[Node]] = []
        centroids_list: List[np.ndarray] = []

        for lbl in unique:
            if lbl == -1:
                # Each noise point becomes its own singleton cluster.
                noise_idx = np.where(labels == -1)[0]
                for i in noise_idx:
                    clusters.append([nodes[i]])
                    centroids_list.append(embeddings[i])
                continue

            idx = np.where(labels == lbl)[0]
            cluster_nodes = [nodes[i] for i in idx]
            clusters.append(cluster_nodes)
            centroids_list.append(embeddings[idx].mean(axis=0))

        centroids = (
            np.vstack(centroids_list)
            if centroids_list
            else np.zeros((0, embeddings.shape[1]))
        )
        return clusters, centroids

    def _maybe_recurse(
        self,
        clusters: List[List[Node]],
        embedding_model_name: str,
        *,
        layer: Optional[int] = None,
    ) -> List[List[Node]]:
        """
        Recursively re-cluster any cluster whose token count exceeds the limit.

        Mirrors RAPTOR_Clustering's recursive behavior. Subclasses can disable
        by setting `recursive_split = False`.
        """
        if not self.recursive_split or self.tokenizer is None:
            return clusters

        out: List[List[Node]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                out.append(cluster)
                continue
            total = sum(self._token_length(n.text) for n in cluster)
            if total <= self.max_length_in_cluster:
                out.append(cluster)
                continue
            if self.verbose:
                logger.info(
                    "%s: recursing on oversized cluster (%d nodes, %d tokens)",
                    self.algorithm_name, len(cluster), total,
                )
            sub_result = self.cluster(cluster, embedding_model_name, layer=layer)
            # If recursion fails to split (one giant cluster), accept it to
            # avoid infinite loops. This mirrors the implicit upstream behavior.
            if len(sub_result.clusters) <= 1:
                out.append(cluster)
            else:
                out.extend(sub_result.clusters)
        return out

    def _token_length(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text.split())  # fallback: word count
        return len(self.tokenizer.encode(text))

    def _compute_metrics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Default metrics. Subclasses override or extend.

        We deliberately do NOT compute silhouette by default — it is O(n^2) and
        will dominate runtime on a long document. Subclasses that want it can
        opt in.
        """
        unique = set(labels.tolist())
        unique.discard(-1)
        n_clusters = len(unique)
        n_noise = int((labels == -1).sum())
        return {
            "n_clusters": float(n_clusters),
            "n_noise": float(n_noise),
            "n_nodes": float(labels.shape[0]),
        }
