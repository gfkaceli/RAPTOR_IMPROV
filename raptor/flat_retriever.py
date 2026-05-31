"""
FlatRetriever: flat vector-similarity baseline for RAPTOR comparison.

This is the standard RAG baseline — chunk the document, embed the chunks,
retrieve the top-k by cosine similarity, pass them to the QA model. No tree,
no clustering, no summarization. Sarthi et al. compared RAPTOR against this
approach and showed that hierarchical indexing substantially outperforms flat
retrieval on long-document QA.

Including this baseline in your evaluation is critical because it answers the
question: "does building a tree help at all, or is flat retrieval good enough?"
If a clustering method produces results worse than flat retrieval, the tree
construction is actively hurting performance.

Interface:
    FlatRetriever implements the same methods as RetrievalAugmentation:
        - add_documents(text)
        - answer_question(question=...)
        - retrieve(question)
        - .tree (mock object with num_layers=0, leaf-only stats)

    This means it drops into the eval loop with one small change:
        # OLD:
        config = factory_fn()
        ra = RetrievalAugmentation(config=config)
        # NEW:
        result = factory_fn()
        ra = result if hasattr(result, 'add_documents') else RetrievalAugmentation(config=result)

Integration:
    1. Drop this file into raptor/flat_retriever.py (NOT in the clustering/ subpackage)
    2. Import in notebook:
           from raptor.flat_retriever import FlatRetriever
    3. Add factory:
           def make_flat_config_v2():
               return FlatRetriever(embedding_model=emb2, qa_model=qa2, top_k=5)
    4. Add to METHODS_V2:
           "flat": (make_flat_config_v2, "Flat SBERT retrieval (no tree)")

Dependencies:
    numpy, sentence-transformers (both already in requirements.txt)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock tree object so the eval loop can read .tree.num_layers etc.
# ---------------------------------------------------------------------------

@dataclass
class _MockNode:
    """Minimal node for the mock tree."""
    index: int
    text: str
    embeddings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _MockTree:
    """
    Quacks like raptor.tree_structures.Tree just enough for the eval loop
    to read tree stats without crashing.
    """
    all_nodes: Dict[int, _MockNode] = field(default_factory=dict)
    leaf_nodes: Dict[int, _MockNode] = field(default_factory=dict)
    num_layers: int = 0
    layer_to_nodes: Dict[int, List[_MockNode]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_tokens: int = 100) -> List[str]:
    """
    Split text into chunks of approximately max_tokens words.

    Uses sentence boundaries to avoid cutting mid-sentence. This replicates
    RAPTOR's chunking behavior (split on sentences, group until hitting the
    token limit) without depending on tiktoken.

    Word count is used as a proxy for token count (~0.75 words per token for
    English prose). This is intentionally the same heuristic RAPTOR uses when
    tiktoken is not available.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())
        if current_len + sentence_len > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = sentence_len
        else:
            current_chunk.append(sentence)
            current_len += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ---------------------------------------------------------------------------
# FlatRetriever
# ---------------------------------------------------------------------------

class FlatRetriever:
    """
    Flat vector-similarity retriever. No tree, no clustering, no summarization.

    Parameters
    ----------
    embedding_model : SBertEmbeddingModel
        The same embedding model used by the RAPTOR methods. This ensures the
        comparison is fair — same embeddings, different retrieval strategy.
    qa_model : BaseQAModel
        The QA model for answer generation.
    top_k : int
        Number of chunks to retrieve per query. Default 5.
    chunk_size : int
        Approximate number of words per chunk. Default 100.
    """

    def __init__(
        self,
        embedding_model,
        qa_model,
        top_k: int = 5,
        chunk_size: int = 100,
    ):
        self.embedding_model = embedding_model
        self.qa_model = qa_model
        self.top_k = top_k
        self.chunk_size = chunk_size

        # Populated by add_documents
        self._chunks: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._tree: Optional[_MockTree] = None

    def add_documents(self, text: str) -> None:
        """Chunk the document and embed all chunks."""
        self._chunks = _chunk_text(text, max_tokens=self.chunk_size)
        if not self._chunks:
            self._embeddings = np.zeros((0, 0))
            self._tree = _MockTree()
            return

        # Embed all chunks using the same model RAPTOR uses
        raw_embeddings = self.embedding_model.create_embedding_batch(self._chunks)
        self._embeddings = np.array(raw_embeddings)

        # Build mock tree for eval loop stats
        mock_nodes = {}
        for i, chunk in enumerate(self._chunks):
            mock_nodes[i] = _MockNode(index=i, text=chunk)

        self._tree = _MockTree(
            all_nodes=mock_nodes,
            leaf_nodes=mock_nodes.copy(),
            num_layers=0,
            layer_to_nodes={0: list(mock_nodes.values())},
        )

        logger.info(
            "FlatRetriever: %d chunks, embedding shape %s",
            len(self._chunks), self._embeddings.shape,
        )

    @property
    def tree(self) -> _MockTree:
        """Mock tree object for eval loop compatibility."""
        if self._tree is None:
            return _MockTree()
        return self._tree

    def retrieve(self, question: str, top_k: Optional[int] = None) -> str:
        """
        Retrieve the top-k most similar chunks by cosine similarity.

        Returns the concatenated text of the top-k chunks, which is what
        RetrievalAugmentation.retrieve() returns.
        """
        if self._embeddings is None or len(self._chunks) == 0:
            return ""

        k = top_k or self.top_k

        # Embed the query
        query_embedding = np.array(
            self.embedding_model.create_embedding(question)
        )

        # Cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = self._embeddings / norms

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            query_normed = query_embedding
        else:
            query_normed = query_embedding / query_norm

        similarities = normed @ query_normed

        # Top-k indices
        k = min(k, len(self._chunks))
        top_indices = np.argsort(similarities)[::-1][:k]

        # Concatenate retrieved chunks
        retrieved_texts = [self._chunks[i] for i in top_indices]
        return "\n\n".join(retrieved_texts)

    def answer_question(self, question: str = "", **kwargs) -> str:
        """Retrieve context and pass to the QA model."""
        # Accept question as kwarg or positional (matches RA interface)
        if not question:
            question = kwargs.get("question", "")
        if not question:
            return ""

        context = self.retrieve(question)
        if not context:
            return ""

        return self.qa_model.answer_question(context, question)

    def __repr__(self) -> str:
        n = len(self._chunks) if self._chunks else 0
        return f"FlatRetriever(chunks={n}, top_k={self.top_k})"
