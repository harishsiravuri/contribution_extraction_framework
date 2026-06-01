"""Minimal R-GCN-style link prediction for "do these two papers share a dataset?".

To avoid the heavy torch_geometric / dgl dependency, we implement a small
multi-relational GCN by hand on top of vanilla torch. For each relation type
we maintain a normalised adjacency matrix (dense — fine for a few thousand
nodes) and apply per-relation linear transforms summed across types. Two
layers, ReLU between, 64-dim hidden.

Pairs are sampled per the spec: 5 random negatives per positive, 80/10/10
split. Returns per-seed test ROC-AUC.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from paper1.downstream.graph_builder import HeteroGraph


@dataclass
class GraphTensors:
    n_nodes: int
    rel_adjs: dict[str, torch.Tensor]
    paper_ids: list[int]


def _normalise(adj: torch.Tensor) -> torch.Tensor:
    deg = adj.sum(dim=1).clamp(min=1.0)
    inv_sqrt = deg.pow(-0.5)
    return adj * inv_sqrt.unsqueeze(0) * inv_sqrt.unsqueeze(1)


def to_tensors(g: HeteroGraph) -> GraphTensors:
    n = g.n_nodes_total
    rel_adjs: dict[str, torch.Tensor] = {}
    for rel, edges in g.edges.items():
        adj = torch.zeros(n, n, dtype=torch.float32)
        for u, v in edges:
            adj[u, v] = 1.0
            adj[v, u] = 1.0  # treat as undirected
        adj += torch.eye(n)
        rel_adjs[rel] = _normalise(adj)
    paper_ids = [g.node_id[("paper", name)] for name in g.nodes.get("paper", [])]
    return GraphTensors(n_nodes=n, rel_adjs=rel_adjs, paper_ids=paper_ids)


class RGCN(nn.Module):
    def __init__(self, n_nodes: int, hidden: int, n_rels: int, n_layers: int = 2) -> None:
        super().__init__()
        self.x = nn.Parameter(torch.randn(n_nodes, hidden) * 0.1)
        self.layers = nn.ModuleList(
            [nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_rels)]) for _ in range(n_layers)]
        )
        self.n_rels = n_rels

    def forward(self, rel_adjs: list[torch.Tensor]) -> torch.Tensor:
        h = self.x
        for li, lyr in enumerate(self.layers):
            agg = torch.zeros_like(h)
            for ri, adj in enumerate(rel_adjs):
                agg = agg + adj @ lyr[ri](h)
            h = agg / max(1, self.n_rels)
            if li < len(self.layers) - 1:
                h = F.relu(h)
        return h


def _build_pairs(g: HeteroGraph, rng: random.Random, neg_per_pos: int = 5) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Build (a, b, label) triples for shared-dataset link prediction."""
    paper_to_datasets: dict[int, set[int]] = {}
    for u, v in g.edges.get("paper_uses_dataset", []):
        paper_to_datasets.setdefault(u, set()).add(v)

    paper_ids = [g.node_id[("paper", n)] for n in g.nodes.get("paper", [])]
    paper_set = set(paper_ids)
    positives: list[tuple[int, int]] = []
    paper_list = [pid for pid in paper_ids if pid in paper_to_datasets]
    if not paper_list:
        return [], [], []
    for i, a in enumerate(paper_list):
        for b in paper_list[i + 1:]:
            if paper_to_datasets[a] & paper_to_datasets[b]:
                positives.append((a, b))

    rng.shuffle(positives)
    if not positives:
        return [], [], []

    # Cap positives if huge
    positives = positives[:5000]

    triples: list[tuple[int, int, int]] = []
    for a, b in positives:
        triples.append((a, b, 1))
        for _ in range(neg_per_pos):
            tries = 0
            while True:
                tries += 1
                c, d = rng.choice(paper_list), rng.choice(paper_list)
                if c == d:
                    continue
                cd = paper_to_datasets.get(c, set())
                dd = paper_to_datasets.get(d, set())
                if not (cd & dd):
                    triples.append((c, d, 0))
                    break
                if tries > 50:
                    triples.append((c, d, 0))  # accept noise
                    break

    rng.shuffle(triples)
    n = len(triples)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    return triples[:train_end], triples[train_end:val_end], triples[val_end:]


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    n_pos = pos.size
    n_neg = neg.size
    # Mann-Whitney U
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, combined.size + 1)
    rank_sum_pos = ranks[:n_pos].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def train_and_eval(
    g: HeteroGraph,
    seed: int = 0,
    hidden: int = 64,
    epochs: int = 30,
    lr: float = 0.005,
) -> dict[str, float | list[float]]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train, val, test = _build_pairs(g, rng)
    if not train or not test:
        return {"test_auc": 0.5, "n_train": 0, "n_test": 0, "train_loss_curve": []}

    gt = to_tensors(g)
    model = RGCN(gt.n_nodes, hidden=hidden, n_rels=len(gt.rel_adjs))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    rel_list = list(gt.rel_adjs.values())

    def _score(h: torch.Tensor, pairs: list[tuple[int, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.tensor([p[0] for p in pairs])
        b = torch.tensor([p[1] for p in pairs])
        y = torch.tensor([p[2] for p in pairs], dtype=torch.float32)
        s = (h[a] * h[b]).sum(dim=1)
        return s, y

    losses: list[float] = []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        h = model(rel_list)
        s, y = _score(h, train)
        loss = F.binary_cross_entropy_with_logits(s, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        h = model(rel_list)
        s, y = _score(h, test)
        scores = s.numpy()
        labels = y.numpy().astype(int)
    test_auc = _roc_auc(scores, labels)
    return {
        "test_auc": test_auc,
        "n_train": len(train),
        "n_test": len(test),
        "train_loss_curve": losses,
        "test_scores": scores.tolist(),
        "test_labels": labels.tolist(),
    }
