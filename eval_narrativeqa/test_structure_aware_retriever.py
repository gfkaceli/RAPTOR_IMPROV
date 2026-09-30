"""
Tests for raptor/structure_aware_retriever.py on a small synthetic tree whose
structure and similarities are known in advance.

Run from the repo root:
    python tests/test_structure_aware_retriever.py
    (or: pytest tests/test_structure_aware_retriever.py)

Modules are loaded by file path so importing raptor/__init__.py (which pulls in
torch, transformers, etc.) is not required to run these tests.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sar = _load("structure_aware_retriever", "raptor/structure_aware_retriever.py")

# Use the genuine RAPTOR Node/Tree classes when available (same fields either way)
if (ROOT / "raptor/tree_structures.py").exists():
    ts = _load("tree_structures", "raptor/tree_structures.py")
    Node, Tree = ts.Node, ts.Tree
else:
    class Node:
        def __init__(self, text, index, children, embeddings):
            self.text, self.index, self.children, self.embeddings = text, index, children, embeddings

    class Tree:
        def __init__(self, all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes):
            self.all_nodes, self.root_nodes, self.leaf_nodes = all_nodes, root_nodes, leaf_nodes
            self.num_layers, self.layer_to_nodes = num_layers, layer_to_nodes


class WhitespaceTokenizer:
    def encode(self, text):
        return text.split()


class DictEmbedder:
    """Returns stored vectors for known texts, or the vector passed as a query."""
    def __init__(self, table):
        self.table = table

    def create_embedding(self, text):
        if isinstance(text, (list, tuple, np.ndarray)):
            return np.asarray(text, dtype=float)
        return np.asarray(self.table[text], dtype=float)


# Leaves: 5 tokens each; parents: 6 tokens; root: 8 tokens
SPEC = {
    0: ("stall market spices sold daily", [1, 0, 0, 0], 0, set()),
    1: ("father left him the stall", [0.9, 0.1, 0, 0], 0, set()),
    2: ("he decides to leave home", [0.7, 0.7, 0, 0], 0, set()),
    3: ("caravan crosses the open desert", [0, 0, 1, 0], 0, set()),
    4: ("a storm scatters the camels", [0, 0, 0.9, 0.1], 0, set()),
    5: ("they finally arrive in east", [0, 0, 0.6, 0.8], 0, set()),
    6: ("Yusuf weighs inheritance against travel dream", [0.8, 0.5, 0, 0], 1, {0, 1, 2}),
    7: ("the hard caravan journey to east", [0, 0, 0.8, 0.6], 1, {3, 4, 5}),
    8: ("a merchant leaves home and journeys east ultimately", [0.5, 0.3, 0.5, 0.4], 2, {6, 7}),
}


def make_tree(spec=SPEC):
    nodes = {i: Node(t, i, set(ch), {"EMB": np.asarray(e, float)})
             for i, (t, e, _, ch) in spec.items()}
    l2n = {}
    for i, (_, _, layer, _) in spec.items():
        l2n.setdefault(layer, []).append(nodes[i])
    max_layer = max(l2n)
    return Tree(nodes, {i: nodes[i] for i in spec if spec[i][2] == max_layer},
                {i: nodes[i] for i in spec if spec[i][2] == 0}, max_layer, l2n)


def make_retriever(tree=None, **kw):
    tree = tree or make_tree()
    table = {nd.text: nd.embeddings["EMB"] for nd in tree.all_nodes.values()}
    return sar.StructureAwareRetriever(tree, DictEmbedder(table), WhitespaceTokenizer(), **kw)


# ---------------------------------------------------------------------------

def test_index_structure():
    r = make_retriever()
    ix = r.index
    assert ix.parents[0] == {6} and ix.parents[6] == {8} and not ix.parents.get(8)
    assert ix.leaf_descendants(8) == [0, 1, 2, 3, 4, 5]
    assert ix.siblings(0) == {1, 2}
    v = ix.validate()
    assert v["problems"] == [] and v["multi_parent_nodes"] == 0
    assert v["nodes_per_layer"] == {0: 6, 1: 2, 2: 1} and v["embedding_key"] == "EMB"


def test_multi_parent_soft_assignment():
    spec = dict(SPEC)
    spec[7] = (spec[7][0], spec[7][1], 1, {2, 3, 4, 5})   # node 2 in two clusters
    r = make_retriever(make_tree(spec))
    assert r.index.parents[2] == {6, 7}
    assert r.index.validate()["multi_parent_nodes"] == 1


def test_scope_policies():
    r = make_retriever(token_budget=100)
    q = [1, 0, 0, 0]
    assert all(l == 0 for l in r.retrieve(q, "leaf_only").layers)
    assert all(l > 0 for l in r.retrieve(q, "summary_only").layers)
    assert set(r.retrieve(q, "collapsed").layers) == {0, 1, 2}


def test_budget_respected_everywhere():
    r = make_retriever(token_budget=17)
    for q in ([1, 0, 0, 0], [0, 0, 1, 0], [0.5, 0.5, 0.5, 0.5], [0.8, 0.5, 0, 0]):
        for p in sar.POLICIES:
            res = r.retrieve(q, p)
            assert res.tokens <= 17, (p, res.tokens)
            assert res.tokens == sum(len(r.index.nodes[n.index].text.split()) for n in res.nodes)


def test_drill_down_grounds_seeded_summary():
    # Query equals P6's vector -> P6 ranks first. Seed limit fits only P6.
    r = make_retriever(token_budget=16, seed_fraction=0.375, expand_per_node=2)
    res = r.retrieve([0.8, 0.5, 0, 0], "drill_down")
    ops = {n.index: n.op for n in res.nodes}
    assert ops[6] == "seed"
    drilled = [i for i, op in ops.items() if op == "drill_down"]
    assert len(drilled) == 2 and set(drilled) <= {0, 1, 2}


def test_roll_up_adds_converging_parent():
    # Two leaves under P7 rank above P7; seed limit fits exactly those two.
    r = make_retriever(token_budget=16, seed_fraction=0.625, rollup_min_children=2)
    res = r.retrieve([0, 0, 0.95, 0.05], "roll_up")
    ops = {n.index: n.op for n in res.nodes}
    assert ops.get(3) == "seed" and ops.get(4) == "seed" and ops.get(7) == "roll_up"


def test_sibling_expansion():
    r = make_retriever(token_budget=15, seed_fraction=0.34, expand_per_node=2)
    res = r.retrieve([1, 0, 0, 0], "sibling")
    ops = {n.index: n.op for n in res.nodes}
    assert ops.get(0) == "seed" and ops.get(1) == "sibling" and ops.get(2) == "sibling"


def test_skip_not_break():
    spec = dict(SPEC)
    spec[0] = (" ".join(["long"] * 50), spec[0][1], 0, set())   # top leaf, 50 tokens
    r = make_retriever(make_tree(spec), token_budget=12)
    res = r.retrieve([1, 0, 0, 0], "leaf_only")
    assert 0 not in [n.index for n in res.nodes] and len(res.nodes) == 2


def test_adaptive_routing():
    r = make_retriever()
    peaked_leaf = {0: 0.99, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.5, 7: 0.45, 8: 0.48}
    flat_leaf = {0: 0.5, 1: 0.49, 2: 0.5, 3: 0.48, 4: 0.5, 5: 0.49, 6: 0.9, 7: 0.1, 8: 0.1}
    assert r._route(peaked_leaf) == "local"
    assert r._route(flat_leaf) == "global"
    res = r.retrieve([1, 0, 0, 0], "adaptive")
    assert res.route in ("local", "global")


def test_structured_order_and_text_format():
    spec = dict(SPEC)
    spec[1] = ("father left\nhim the stall", spec[1][1], 0, set())
    r = make_retriever(make_tree(spec), token_budget=100)
    res = r.retrieve([1, 0, 0, 0], "collapsed")
    layers = res.layers
    first_leaf = layers.index(0)
    assert all(l > 0 for l in layers[:first_leaf]) and all(l == 0 for l in layers[first_leaf:])
    leaf_ids = [n.index for n in res.nodes if n.layer == 0]
    assert leaf_ids == sorted(leaf_ids)                      # document order
    assert "father left him the stall\n\n" in res.context   # RAPTOR get_text format


def test_determinism_and_dim_check():
    r = make_retriever(token_budget=17)
    a = r.retrieve([0.5, 0.5, 0.5, 0.5], "drill_down")
    b = r.retrieve([0.5, 0.5, 0.5, 0.5], "drill_down")
    assert [n.index for n in a.nodes] == [n.index for n in b.nodes] and a.context == b.context
    try:
        r.retrieve([1, 0, 0], "collapsed")
        raise AssertionError("expected a dimension-mismatch error")
    except ValueError:
        pass


def test_embedding_consistency_check():
    r = make_retriever()
    assert sar.verify_embedding_consistency(r)["ok"]
    rng = np.random.default_rng(0)
    r.embedding_model = DictEmbedder({t: rng.normal(size=4) for t in r.embedding_model.table})
    assert not sar.verify_embedding_consistency(r)["ok"]


def test_matches_raptor_collapsed_ranking():
    """Reimplements RAPTOR's retrieve_information_collapse_tree ranking
    (scipy cosine distance, ascending argsort) and checks top-k agreement."""
    from scipy import spatial

    r = make_retriever()
    tree = r.index.tree

    class _RaptorRetriever:
        def retrieve_information_collapse_tree(self, query, top_k, max_tokens):
            q = np.asarray(query, float)
            node_list = [tree.all_nodes[i] for i in sorted(tree.all_nodes)]
            d = [spatial.distance.cosine(q, n.embeddings["EMB"]) for n in node_list]
            return [node_list[i] for i in np.argsort(d)[:top_k]], ""

    class _RA:
        retriever = _RaptorRetriever()

    queries = [[1, 0, 0, 0], [0, 0.3, 0.9, 0.2], [0.4, 0.4, 0.4, 0.7], [0.8, 0.5, 0, 0]]
    out = sar.verify_against_raptor(r, _RA(), queries, k=5)
    assert out["mean_overlap"] == 1.0, out


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
