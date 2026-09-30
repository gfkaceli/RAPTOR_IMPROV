"""
structure_aware_retriever.py — Structure-aware retrieval over existing RAPTOR trees.

RAPTOR's TreeRetriever offers two modes. Collapsed-tree retrieval ranks every
node by cosine similarity and never reads Node.children, so the tree topology is
invisible at query time. Tree traversal descends greedily from the root and
adds top_k nodes per layer with no token budget. This module adds retrieval
policies in which the tree's edges (parent, child, sibling) decide what enters
the context, under ONE fixed token budget shared by every policy.

Nothing here modifies the tree or requires rebuilding it. RAPTOR's Tree already
stores layer_to_nodes, and each Node stores its children, so the parent map is
obtained by inverting children. The module depends only on numpy and works on
any object exposing the upstream Tree/Node fields:
    Tree: all_nodes (dict index->Node), layer_to_nodes (dict layer->[Node]),
          num_layers
    Node: index, text, children (set of int), embeddings (dict key->vector)

Policies (all share the same token budget and admission rule):
    collapsed     Top nodes by similarity from all layers (baseline).
    leaf_only     Top leaves only (within-tree analogue of flat retrieval).
    summary_only  Top summary nodes only (layers >= 1). Diagnostic.
    drill_down    Seed with collapsed ranking; for each seeded summary, add its
                  most query-similar leaf descendants (grounds abstractions in
                  source text).
    roll_up       Seed; add any parent with >= rollup_min_children seeded
                  children (one summary covering converging evidence).
    sibling       Seed; add the most query-similar siblings of seeded nodes
                  (evidence split across neighbouring clusters).
    adaptive      Route per query: "local" -> leaf_only, "global" -> roll_up,
                  using a calibration-robust peakedness test (see _route).

Admission rule. Candidates are admitted greedily in priority order while the
token total stays within the limit. A candidate that does not fit is SKIPPED,
not a stopping point (RAPTOR's collapsed loop breaks at the first overflow).
Expansion policies admit seeds up to seed_fraction * budget, then expansion
candidates, then backfill any remaining budget with the similarity ranking, so
every policy uses the budget comparably and differences reflect composition,
not context volume.

Context text uses RAPTOR's get_text format (each node's lines joined by spaces,
nodes separated by a blank line). With context_order="structured" (default),
summaries come first (higher layers first) and leaves follow in document order;
"similarity" orders purely by score, as RAPTOR does.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

FIXED_POLICIES = ("collapsed", "leaf_only", "summary_only",
                  "drill_down", "roll_up", "sibling")
POLICIES = FIXED_POLICIES + ("adaptive",)


# ---------------------------------------------------------------------------
# Structure index
# ---------------------------------------------------------------------------

def resolve_embedding_key(tree, preferred: str = "EMB") -> str:
    """Return the embedding key stored on the tree's nodes.

    RAPTOR stores node embeddings as {model_key: vector}. When the config is
    built with embedding_model=..., the key is "EMB". Falls back to the single
    stored key if the preferred one is absent; raises if ambiguous.
    """
    node = next(iter(tree.all_nodes.values()))
    keys = list(node.embeddings.keys())
    if preferred in node.embeddings:
        return preferred
    if len(keys) == 1:
        return keys[0]
    raise KeyError(f"Embedding key '{preferred}' not found; node has {keys}. "
                   f"Pass embedding_key explicitly.")


class StructureIndex:
    """Layer, parent, child, and embedding lookups for one RAPTOR tree."""

    def __init__(self, tree, embedding_key: str = "EMB"):
        self.tree = tree
        self.nodes: Dict[int, object] = dict(tree.all_nodes)
        self.embedding_key = resolve_embedding_key(tree, embedding_key)

        # Layer map from layer_to_nodes (same source as RAPTOR's reverse_mapping)
        self.layer_of: Dict[int, int] = {}
        for layer, nodes in tree.layer_to_nodes.items():
            for nd in nodes:
                self.layer_of[nd.index] = int(layer)
        self.max_layer = max(self.layer_of.values()) if self.layer_of else 0

        # Children and inverted parents (one-to-many: soft clustering can give
        # a node several parents)
        self.children: Dict[int, Set[int]] = {
            i: set(int(c) for c in (nd.children or ())) for i, nd in self.nodes.items()
        }
        self.parents: Dict[int, Set[int]] = defaultdict(set)
        for p, kids in self.children.items():
            for c in kids:
                self.parents[c].add(p)

        # Embedding matrix aligned with self.order, L2-normalized for cosine
        self.order = np.array(sorted(self.nodes), dtype=int)
        self.pos = {int(i): k for k, i in enumerate(self.order)}
        E = np.vstack([np.asarray(self.nodes[int(i)].embeddings[self.embedding_key],
                                  dtype=float).ravel() for i in self.order])
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.E = E / norms
        self.dim = self.E.shape[1]

        self.is_leaf = np.array([self.layer_of.get(int(i), 0) == 0 for i in self.order])
        self._leaf_desc: Dict[int, List[int]] = {}

    # --- traversal helpers --------------------------------------------------

    def layer(self, idx: int) -> int:
        return self.layer_of.get(idx, 0)

    def leaf_descendants(self, idx: int) -> List[int]:
        """All leaf nodes under idx (memoized, iterative DFS)."""
        if idx in self._leaf_desc:
            return self._leaf_desc[idx]
        if self.layer(idx) == 0:
            return [idx]
        out, stack, seen = set(), [idx], set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            for c in self.children.get(n, ()):
                if self.layer(c) == 0:
                    out.add(c)
                else:
                    stack.append(c)
        self._leaf_desc[idx] = sorted(out)
        return self._leaf_desc[idx]

    def siblings(self, idx: int) -> Set[int]:
        sib = set()
        for p in self.parents.get(idx, ()):
            sib |= self.children.get(p, set())
        sib.discard(idx)
        return sib

    # --- validation ---------------------------------------------------------

    def validate(self) -> Dict:
        """Structural checks. Problems are listed; nothing is modified."""
        problems = []
        missing = {c for kids in self.children.values() for c in kids
                   if c not in self.nodes}
        if missing:
            problems.append(f"{len(missing)} child indices reference missing nodes")
        no_layer = [i for i in self.nodes if i not in self.layer_of]
        if no_layer:
            problems.append(f"{len(no_layer)} nodes absent from layer_to_nodes")
        leaves = [i for i in self.nodes if self.layer(i) == 0]
        orphan_leaves = [i for i in leaves if not self.parents.get(i)]
        if self.max_layer > 0 and orphan_leaves:
            problems.append(f"{len(orphan_leaves)} leaves have no parent")
        bad_edges = [(p, c) for p, kids in self.children.items() for c in kids
                     if c in self.nodes and self.layer(c) >= self.layer(p)]
        if bad_edges:
            problems.append(f"{len(bad_edges)} edges do not point to a lower layer")
        multi = sum(1 for c, ps in self.parents.items() if len(ps) > 1)
        per_layer = defaultdict(int)
        for i in self.nodes:
            per_layer[self.layer(i)] += 1
        return {
            "n_nodes": len(self.nodes),
            "n_leaves": len(leaves),
            "max_layer": self.max_layer,
            "nodes_per_layer": dict(sorted(per_layer.items())),
            "multi_parent_nodes": multi,
            "orphan_leaves": len(orphan_leaves),
            "embedding_key": self.embedding_key,
            "embedding_dim": self.dim,
            "problems": problems,
        }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@dataclass
class RetrievedNode:
    index: int
    layer: int
    score: float
    op: str          # seed | drill_down | roll_up | sibling
    tokens: int


@dataclass
class RetrievalResult:
    policy: str
    context: str
    nodes: List[RetrievedNode] = field(default_factory=list)
    tokens: int = 0
    budget: int = 0
    route: Optional[str] = None        # adaptive only: local | global
    fallback: bool = False             # summary_only on a tree without summaries

    @property
    def layers(self) -> List[int]:
        return [n.layer for n in self.nodes]

    @property
    def ops(self) -> List[str]:
        return [n.op for n in self.nodes]


class StructureAwareRetriever:
    def __init__(self, tree, embedding_model, tokenizer, embedding_key: str = "EMB",
                 token_budget: int = 1400, seed_fraction: float = 0.6,
                 expand_per_node: int = 2, rollup_min_children: int = 2,
                 adaptive_margin: float = 0.0, context_order: str = "structured"):
        if not 0 < seed_fraction <= 1:
            raise ValueError("seed_fraction must be in (0, 1]")
        if context_order not in ("structured", "similarity"):
            raise ValueError("context_order must be 'structured' or 'similarity'")
        self.index = StructureIndex(tree, embedding_key)
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.token_budget = int(token_budget)
        self.seed_fraction = float(seed_fraction)
        self.expand_per_node = int(expand_per_node)
        self.rollup_min_children = int(rollup_min_children)
        self.adaptive_margin = float(adaptive_margin)
        self.context_order = context_order
        self._tok_cache: Dict[int, int] = {}

    @classmethod
    def from_retrieval_augmentation(cls, ra, **kwargs):
        """Build from a loaded RetrievalAugmentation, reusing the exact
        embedding model, tokenizer, and embedding key its TreeRetriever uses."""
        r = ra.retriever
        return cls(ra.tree, r.embedding_model, r.tokenizer,
                   embedding_key=r.context_embedding_model, **kwargs)

    # --- primitives ---------------------------------------------------------

    def _ntok(self, idx: int) -> int:
        if idx not in self._tok_cache:
            self._tok_cache[idx] = len(self.tokenizer.encode(self.index.nodes[idx].text))
        return self._tok_cache[idx]

    def _scores(self, query: str) -> Dict[int, float]:
        q = np.asarray(self.embedding_model.create_embedding(query), dtype=float).ravel()
        if q.shape[0] != self.index.dim:
            raise ValueError(f"Query embedding dim {q.shape[0]} != tree embedding dim "
                             f"{self.index.dim}: the embedding model does not match "
                             f"the one used to build this tree.")
        n = np.linalg.norm(q)
        q = q / (n if n else 1.0)
        sims = self.index.E @ q
        return {int(i): float(s) for i, s in zip(self.index.order, sims)}

    @staticmethod
    def _ranked(scores: Dict[int, float], pool: Sequence[int]) -> List[int]:
        # Deterministic: score desc, then index asc for ties
        return sorted(pool, key=lambda i: (-scores[i], i))

    def _admit(self, candidates, op, scores, selected, used, limit):
        for idx in candidates:
            if idx in selected:
                continue
            t = self._ntok(idx)
            if used + t > limit:
                continue  # skip, keep scanning for a node that fits
            selected[idx] = RetrievedNode(idx, self.index.layer(idx), scores[idx], op, t)
            used += t
        return used

    def _route(self, scores: Dict[int, float]) -> str:
        """Local vs global, robust to leaf/summary similarity calibration.

        Raw leaf and summary similarities are not on a common scale (summaries
        are longer and more abstract), so comparing best leaf with best summary
        directly is biased. Instead compare how strongly the best node stands
        out within its own layer group (a z-score). A sharply peaked leaf
        distribution signals a local question.
        """
        leaves = [s for i, s in scores.items() if self.index.layer(i) == 0]
        summ = [s for i, s in scores.items() if self.index.layer(i) > 0]
        if not summ:
            return "local"

        def peak(v):
            v = np.asarray(v, dtype=float)
            sd = v.std()
            return (v.max() - v.mean()) / sd if sd > 1e-9 else 0.0

        return "local" if peak(leaves) - peak(summ) >= self.adaptive_margin else "global"

    def _render(self, selected: Dict[int, RetrievedNode]) -> List[RetrievedNode]:
        nodes = list(selected.values())
        if self.context_order == "similarity":
            return sorted(nodes, key=lambda n: (-n.score, n.index))
        summaries = sorted([n for n in nodes if n.layer > 0], key=lambda n: (-n.layer, n.index))
        leaves = sorted([n for n in nodes if n.layer == 0], key=lambda n: n.index)
        return summaries + leaves

    def _text(self, nodes: List[RetrievedNode]) -> str:
        # Identical formatting to raptor.utils.get_text
        return "".join(f"{' '.join(self.index.nodes[n.index].text.splitlines())}\n\n"
                       for n in nodes)

    # --- policies -----------------------------------------------------------

    def retrieve(self, query: str, policy: str = "collapsed") -> RetrievalResult:
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy '{policy}'. Choices: {POLICIES}")
        scores = self._scores(query)
        idx = self.index
        all_ranked = self._ranked(scores, list(scores))
        B = self.token_budget
        selected: Dict[int, RetrievedNode] = {}
        route, fallback = None, False

        if policy == "adaptive":
            route = self._route(scores)
            policy_eff = "leaf_only" if route == "local" else "roll_up"
        else:
            policy_eff = policy

        if policy_eff == "collapsed":
            self._admit(all_ranked, "seed", scores, selected, 0, B)

        elif policy_eff == "leaf_only":
            self._admit([i for i in all_ranked if idx.layer(i) == 0], "seed",
                        scores, selected, 0, B)

        elif policy_eff == "summary_only":
            summ = [i for i in all_ranked if idx.layer(i) > 0]
            if summ:
                self._admit(summ, "seed", scores, selected, 0, B)
            else:
                fallback = True
                self._admit(all_ranked, "seed", scores, selected, 0, B)

        else:  # expansion policies
            seed_limit = int(B * self.seed_fraction)
            used = self._admit(all_ranked, "seed", scores, selected, 0, seed_limit)
            seeds = [i for i in all_ranked if i in selected]  # in rank order
            expansions: List[int] = []

            if policy_eff == "drill_down":
                for s in seeds:
                    if idx.layer(s) == 0:
                        continue
                    desc = [d for d in idx.leaf_descendants(s) if d not in selected]
                    expansions += self._ranked(scores, desc)[: self.expand_per_node]
                op = "drill_down"

            elif policy_eff == "roll_up":
                hits = defaultdict(int)
                for s in seeds:
                    for p in idx.parents.get(s, ()):
                        hits[p] += 1
                parents = [p for p, h in hits.items()
                           if h >= self.rollup_min_children and p not in selected]
                expansions = self._ranked(scores, parents)
                op = "roll_up"

            elif policy_eff == "sibling":
                for s in seeds:
                    sib = [x for x in idx.siblings(s) if x not in selected]
                    expansions += self._ranked(scores, sib)[: self.expand_per_node]
                op = "sibling"

            used = self._admit(expansions, op, scores, selected, used, B)
            self._admit(all_ranked, "seed", scores, selected, used, B)  # backfill

        nodes = self._render(selected)
        return RetrievalResult(policy=policy, context=self._text(nodes), nodes=nodes,
                               tokens=sum(n.tokens for n in nodes), budget=B,
                               route=route, fallback=fallback)


# ---------------------------------------------------------------------------
# Sanity checks to run on real trees before trusting results
# ---------------------------------------------------------------------------

def verify_embedding_consistency(retriever: StructureAwareRetriever, n: int = 3,
                                 min_cosine: float = 0.99) -> Dict:
    """Re-embed a few leaves with the current embedding model and compare with
    the stored vectors. Cosine near 1.0 confirms the query-side model matches
    the model that built the tree; a same-dimension model mismatch would
    otherwise pass silently and corrupt every retrieval."""
    idx = retriever.index
    leaves = [int(i) for i in idx.order if idx.layer(int(i)) == 0][:n]
    cos = []
    for i in leaves:
        v = np.asarray(retriever.embedding_model.create_embedding(idx.nodes[i].text),
                       dtype=float).ravel()
        v = v / (np.linalg.norm(v) or 1.0)
        cos.append(float(idx.E[idx.pos[i]] @ v))
    return {"cosines": [round(c, 4) for c in cos],
            "ok": bool(cos) and min(cos) >= min_cosine}


def verify_against_raptor(retriever: StructureAwareRetriever, ra,
                          queries: Sequence[str], k: int = 10) -> Dict:
    """Check that this module's similarity ranking reproduces RAPTOR's own
    collapsed-tree ranking (top-k node indices, ignoring token limits)."""
    overlaps = []
    for q in queries:
        scores = retriever._scores(q)
        ours = retriever._ranked(scores, list(scores))[:k]
        nodes, _ = ra.retriever.retrieve_information_collapse_tree(q, k, 10 ** 9)
        theirs = [n.index for n in nodes]
        overlaps.append(len(set(ours) & set(theirs)) / max(len(theirs), 1))
    return {"mean_overlap": round(float(np.mean(overlaps)), 4) if overlaps else None,
            "per_query": [round(o, 4) for o in overlaps]}
