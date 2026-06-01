"""Build a heterogeneous-ish graph from a directory of ContributionRecord JSONs.

Node types: Paper, Method, Dataset, Task, Metric.
Edge types: paper-uses-method, paper-uses-dataset, paper-on-task,
            paper-reports-metric, dataset-supports-task.

For Phase 7 we keep this simple: one undirected edge list per type, indexed
into a flat node id space. The downstream R-GCN treats edge-types as separate
adjacency matrices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from paper1.schema import ContributionRecord
from paper1.voting import _norm


@dataclass
class HeteroGraph:
    nodes: dict[str, list[str]] = field(default_factory=dict)  # node_type -> ordered names
    node_id: dict[tuple[str, str], int] = field(default_factory=dict)  # (type, name) -> id
    edges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)  # edge_type -> list of (u, v)
    n_nodes_total: int = 0

    def add_node(self, node_type: str, name: str) -> int:
        key = (node_type, name)
        if key in self.node_id:
            return self.node_id[key]
        idx = self.n_nodes_total
        self.node_id[key] = idx
        self.nodes.setdefault(node_type, []).append(name)
        self.n_nodes_total += 1
        return idx

    def add_edge(self, edge_type: str, u: int, v: int) -> None:
        self.edges.setdefault(edge_type, []).append((u, v))


def build_graph(by_paper_dir: Path) -> HeteroGraph:
    g = HeteroGraph()
    for path in sorted(by_paper_dir.glob("*.json")):
        try:
            rec = ContributionRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not rec.contributions:
            continue
        paper_id = rec.paper_id
        p_idx = g.add_node("paper", paper_id)
        for c in rec.contributions:
            method = _norm(c.method.name)
            task = _norm(c.task.name)
            if method:
                m_idx = g.add_node("method", method)
                g.add_edge("paper_uses_method", p_idx, m_idx)
            if task:
                t_idx = g.add_node("task", task)
                g.add_edge("paper_on_task", p_idx, t_idx)
            for d in c.datasets:
                dn = _norm(d.name)
                if not dn:
                    continue
                d_idx = g.add_node("dataset", dn)
                g.add_edge("paper_uses_dataset", p_idx, d_idx)
                if task:
                    g.add_edge("dataset_supports_task", d_idx, t_idx)
            for m in c.metrics:
                mn = _norm(m.name)
                if mn:
                    mt_idx = g.add_node("metric", mn)
                    g.add_edge("paper_reports_metric", p_idx, mt_idx)
    return g


def stats(g: HeteroGraph) -> dict:
    return {
        "n_nodes_total": g.n_nodes_total,
        "n_nodes_by_type": {k: len(v) for k, v in g.nodes.items()},
        "n_edges_by_type": {k: len(v) for k, v in g.edges.items()},
    }


def to_json(g: HeteroGraph) -> dict:
    return {
        "nodes": g.nodes,
        "edges": g.edges,
    }
