import os
import sys
import types
from typing import Dict, List, Set
import openai
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# -----------------------------------------------------------------------------
# Compatibility shims so the demo can run without external services/packages.
# -----------------------------------------------------------------------------
if "openai" not in sys.modules:
    sys.modules["openai"] = types.ModuleType("openai")

if "tenacity" not in sys.modules:
    tenacity = types.ModuleType("tenacity")
    tenacity.retry = lambda *a, **k: (lambda f: f)
    tenacity.stop_after_attempt = lambda *a, **k: None
    tenacity.wait_random_exponential = lambda *a, **k: None
    sys.modules["tenacity"] = tenacity

if "tiktoken" not in sys.modules:
    tiktoken = types.ModuleType("tiktoken")

    class _SimpleEncoding:
        def encode(self, text: str):
            return text.split()

    tiktoken.get_encoding = lambda _name: _SimpleEncoding()
    sys.modules["tiktoken"] = tiktoken

# Stub modules expected by tree_builders/tree_retrievers
EmbeddingModels = types.ModuleType("EmbeddingModels")
SummarizationModels = types.ModuleType("SummarizationModels")


class BaseEmbeddingModel:
    def create_embedding(self, text):
        raise NotImplementedError


class DemoEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        self.vocab = [
            "raptor", "tree", "retrieval", "embedding", "summary", "question",
            "chunk", "node", "cluster", "faiss", "document", "context",
            "answer", "hierarchy", "search", "token",
        ]

    def create_embedding(self, text):
        text_l = text.lower()
        counts = np.array([text_l.count(term) for term in self.vocab], dtype=np.float32)
        extras = np.array([
            len(text_l.split()), len(set(text_l.split())), text_l.count("."), text_l.count("?")
        ], dtype=np.float32)
        vec = np.concatenate([counts, extras])
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm else vec.astype(np.float32)


class OpenAIEmbeddingModel(DemoEmbeddingModel):
    pass


class BaseSummarizationModel:
    def summarize(self, context, max_tokens=150):
        raise NotImplementedError


class DemoSummarizationModel(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        sentences = [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]
        summary = ". ".join(sentences[:2])
        if summary and not summary.endswith("."):
            summary += "."
        return " ".join(summary.split()[:max_tokens])


class GPT3TurboSummarizationModel(DemoSummarizationModel):
    pass


EmbeddingModels.BaseEmbeddingModel = BaseEmbeddingModel
EmbeddingModels.OpenAIEmbeddingModel = OpenAIEmbeddingModel
SummarizationModels.BaseSummarizationModel = BaseSummarizationModel
SummarizationModels.GPT3TurboSummarizationModel = GPT3TurboSummarizationModel

sys.modules["EmbeddingModels"] = EmbeddingModels
sys.modules["SummarizationModels"] = SummarizationModels

# Import the uploaded modules that are directly runnable with the shims above.
from raptor.utils import split_text
from raptor.tree_builders import TreeBuilder, TreeBuilderConfig
from raptor.tree_retrievers import TreeRetriever, TreeRetrieverConfig
from raptor.tree_structures import Node, Tree


class DemoTreeBuilder(TreeBuilder):
    """Concrete builder for this demo.

    The uploaded tree_builders.py supplies the shared logic, but the actual
    cluster-based implementation file is only a stub. This class demonstrates the
    intended behavior by merging neighboring nodes pairwise and summarizing them.
    """

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        current_nodes = current_level_nodes
        next_node_index = max(all_tree_nodes.keys()) + 1 if all_tree_nodes else 0

        for layer in range(1, self.num_layers + 1):
            ordered_nodes = [current_nodes[idx] for idx in sorted(current_nodes.keys())]
            if len(ordered_nodes) <= 1:
                break

            new_level_nodes: Dict[int, Node] = {}
            for i in range(0, len(ordered_nodes), 2):
                children = ordered_nodes[i:i + 2]
                child_ids: Set[int] = {node.index for node in children}
                combined_text = "\n".join(node.text for node in children)
                summary = self.summarize(combined_text, max_tokens=self.summarization_length)
                idx, parent = self.create_node(next_node_index, summary, child_ids)
                new_level_nodes[idx] = parent
                all_tree_nodes[idx] = parent
                next_node_index += 1

            layer_to_nodes[layer] = list(new_level_nodes.values())
            current_nodes = new_level_nodes
            if len(current_nodes) == 1:
                break

        return current_nodes


class LocalFlatRetriever:
    """Small local retriever that mirrors the intent of FaissRetriever.py.

    The uploaded FaissRetriever.py is conceptually a flat vector retriever over
    chunks. Its current file uses package-relative imports that make it awkward to
    run standalone, so this demo implements the same flow locally: split -> embed
    -> rank by inner product.
    """

    def __init__(self, embedding_model: DemoEmbeddingModel, max_tokens: int = 25, top_k: int = 3):
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.tokenizer = sys.modules["tiktoken"].get_encoding("cl100k_base")
        self.chunks: List[str] = []
        self.embeddings: np.ndarray | None = None

    def build_from_text(self, text: str):
        self.chunks = split_text(text, self.tokenizer, self.max_tokens)
        self.embeddings = np.array([self.embedding_model.create_embedding(c) for c in self.chunks], dtype=np.float32)

    def retrieve(self, query: str) -> str:
        q = self.embedding_model.create_embedding(query)
        scores = self.embeddings @ q
        order = np.argsort(-scores)[: self.top_k]
        return "\n\n".join(self.chunks[i] for i in order)


class DemoQAModel:
    def answer_question(self, context: str, question: str) -> str:
        q_terms = set(question.lower().replace("?", "").split())
        best = ""
        best_score = -1
        for sentence in [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]:
            score = len(q_terms.intersection(sentence.lower().split()))
            if score > best_score:
                best = sentence
                best_score = score
        return best or context[:200]


def print_header(title: str):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def show_chunks(text: str):
    print_header("1) utils.split_text: chunk the source document")
    tokenizer = sys.modules["tiktoken"].get_encoding("cl100k_base")
    chunks = split_text(text, tokenizer=tokenizer, max_tokens=30)
    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i}: {chunk}")
    return chunks


def build_tree(text: str):
    print_header("2) tree_builders.py scaffold: create leaf nodes and parent summaries")
    embedding_model = DemoEmbeddingModel()
    config = TreeBuilderConfig(
        max_tokens=30,
        num_layers=3,
        summarization_length=30,
        summarization_model=DemoSummarizationModel(),
        embedding_models={"demo": embedding_model},
        cluster_embedding_model="demo",
    )
    builder = DemoTreeBuilder(config)
    tree = builder.build_from_text(text, use_multithreading=False)

    print(f"Total nodes: {len(tree.all_nodes)}")
    print(f"Leaf nodes:  {len(tree.leaf_nodes)}")
    print(f"Root nodes:  {len(tree.root_nodes)}")
    print(f"Layers:      {sorted(tree.layer_to_nodes.keys())}")
    for layer, nodes in tree.layer_to_nodes.items():
        print(f"  Layer {layer}: {len(nodes)} node(s)")
        for node in nodes[:2]:
            preview = node.text[:100].replace("\n", " ")
            print(f"    - Node {node.index}: {preview}")
    return tree, embedding_model


def run_tree_retrieval(tree: Tree, embedding_model: DemoEmbeddingModel):
    print_header("3) tree_retrievers.py: hierarchical retrieval over the built tree")
    config = TreeRetrieverConfig(
        top_k=2,
        selection_mode="top_k",
        context_embedding_model="demo",
        embedding_model=embedding_model,
        start_layer=max(tree.layer_to_nodes.keys()),
        num_layers=min(2, max(tree.layer_to_nodes.keys()) + 1),
    )
    retriever = TreeRetriever(config, tree)
    query = "How are summaries and embeddings used in retrieval?"
    context, layer_info = retriever.retrieve(
        query,
        top_k=2,
        max_tokens=160,
        collapse_tree=False,
        return_layer_information=True,
    )
    print("Query:", query)
    print("Selected nodes with layers:", layer_info)
    print("Retrieved context:\n", context[:700])
    print("Answer-like output:", DemoQAModel().answer_question(context, query))


def run_flat_retrieval(text: str, embedding_model: DemoEmbeddingModel):
    print_header("4) FaissRetriever.py concept: flat chunk retrieval")
    retriever = LocalFlatRetriever(embedding_model=embedding_model, max_tokens=25, top_k=3)
    retriever.build_from_text(text)
    query = "Which chunks describe FAISS, search, and chunk embeddings?"
    context = retriever.retrieve(query)
    print("Query:", query)
    print("Retrieved context:\n", context[:700])


def main():
    demo_text = (
        "RAPTOR-style systems begin by splitting a long document into manageable chunks. "
        "Each chunk becomes a leaf node that stores raw text and one or more embeddings. "
        "A tree builder then summarizes groups of child nodes into parent nodes, producing a hierarchy of compressed representations. "
        "This hierarchy can make retrieval cheaper because upper layers provide compact summaries before the system drills down. "
        "A tree retriever starts from some layer, scores nodes against the query embedding, and follows the most relevant branches. "
        "The retriever can also collapse the hierarchy and compare the query against all nodes directly. "
        "FAISS retrieval is the flat baseline: it embeds chunks, indexes them, and returns the nearest chunks for a query. "
        "Question answering uses the retrieved context, while summarization is mainly used during tree construction. "
        "Utility helpers support token-based splitting, distance calculations, node traversal, and text concatenation."
    )

    show_chunks(demo_text)
    tree, embedding_model = build_tree(demo_text)
    run_tree_retrieval(tree, embedding_model)
    run_flat_retrieval(demo_text, embedding_model)

    print_header("5) Notes")
    print(
        "This demo intentionally uses local stand-ins for external models and packages. "
        "That keeps it runnable while still showing the relationships among chunking, embedding, summarization, "
        "hierarchical tree construction, tree-based retrieval, and flat vector retrieval."
    )


if __name__ == "__main__":
    main()
