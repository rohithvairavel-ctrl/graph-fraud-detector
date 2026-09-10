"""Build payment graph and compute network features for fraud detection."""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

try:
    import community as community_louvain  # python-louvain
except ImportError:  # pragma: no cover
    community_louvain = None


def build_graph(transfers: pd.DataFrame) -> nx.DiGraph:
    """Nodes = accounts, directed edges = transfers (aggregated weight)."""
    g = nx.DiGraph()
    grouped = (
        transfers.groupby(["src_account", "dst_account"])
        .agg(weight=("amount", "sum"), n_tx=("tx_id", "count"))
        .reset_index()
    )
    for _, row in grouped.iterrows():
        g.add_edge(
            row["src_account"],
            row["dst_account"],
            weight=float(row["weight"]),
            n_tx=int(row["n_tx"]),
        )
    return g


def _louvain_communities(g_undirected: nx.Graph) -> dict[Any, int]:
    if community_louvain is not None:
        return community_louvain.best_partition(g_undirected, random_state=42)
    mapping: dict[Any, int] = {}
    for i, comp in enumerate(nx.connected_components(g_undirected)):
        for n in comp:
            mapping[n] = i
    return mapping


def _triangle_counts(g_undirected: nx.Graph) -> dict[Any, int]:
    return nx.triangles(g_undirected)


def _accounts_in_simple_cycles(g: nx.DiGraph, max_nodes: int = 1500) -> set[Any]:
    """Flag nodes in non-trivial strongly connected components (cycle participation).

    Uses SCC instead of enumerating simple_cycles (exponential) for scalability.
    """
    flagged: set[Any] = set()
    for scc in nx.strongly_connected_components(g):
        if len(scc) >= 3:
            flagged.update(scc)
        elif len(scc) == 2:
            flagged.update(scc)
        else:
            n = next(iter(scc))
            if g.has_edge(n, n):
                flagged.add(n)
    return flagged


def compute_graph_features(
    transfers: pd.DataFrame,
    account_features: pd.DataFrame,
) -> pd.DataFrame:
    """Compute degree, PageRank, Louvain community, shared device/IP, cycles, motifs."""
    g = build_graph(transfers)
    for aid in account_features["account_id"]:
        if aid not in g:
            g.add_node(aid)

    g_u = g.to_undirected()

    degree = dict(g_u.degree())
    in_deg = dict(g.in_degree())
    out_deg = dict(g.out_degree())
    pagerank = nx.pagerank(g, alpha=0.85, weight="weight")
    clustering = nx.clustering(g_u)
    communities = _louvain_communities(g_u)
    triangles = _triangle_counts(g_u)
    cycle_nodes = _accounts_in_simple_cycles(g)

    comp_size: dict[Any, int] = {}
    for comp in nx.connected_components(g_u):
        s = len(comp)
        for n in comp:
            comp_size[n] = s

    from collections import Counter

    comm_counts = Counter(communities.values())
    community_size = {n: comm_counts[c] for n, c in communities.items()}

    device_counts = account_features.groupby("device_id")["account_id"].transform("count")
    ip_counts = account_features.groupby("ip_id")["account_id"].transform("count")
    shared_device = dict(zip(account_features["account_id"], device_counts - 1))
    shared_ip = dict(zip(account_features["account_id"], ip_counts - 1))

    night_map = dict(zip(account_features["account_id"], account_features["night_tx_ratio"]))
    neighbor_proxy: dict[Any, float] = {}
    for n in g.nodes():
        nbrs = list(g_u.neighbors(n))
        if not nbrs:
            neighbor_proxy[n] = 0.0
        else:
            neighbor_proxy[n] = float(np.mean([night_map.get(x, 0.0) for x in nbrs]))

    rows = []
    for aid in account_features["account_id"]:
        rows.append(
            {
                "account_id": aid,
                "degree": degree.get(aid, 0),
                "in_degree": in_deg.get(aid, 0),
                "out_degree": out_deg.get(aid, 0),
                "pagerank": pagerank.get(aid, 0.0),
                "clustering": clustering.get(aid, 0.0),
                "community_id": communities.get(aid, -1),
                "community_size": community_size.get(aid, 1),
                "component_size": comp_size.get(aid, 1),
                "shared_device_count": int(shared_device.get(aid, 0)),
                "shared_ip_count": int(shared_ip.get(aid, 0)),
                "triangle_count": triangles.get(aid, 0),
                "cycle_flag": 1 if aid in cycle_nodes else 0,
                "neighbor_fraud_ratio_proxy": neighbor_proxy.get(aid, 0.0),
            }
        )
    return pd.DataFrame(rows)


def neighborhood_subgraph(
    transfers: pd.DataFrame,
    account_id: str,
    hops: int = 1,
) -> nx.DiGraph:
    """Extract k-hop neighborhood around an account for visualization."""
    g = build_graph(transfers)
    if account_id not in g:
        g.add_node(account_id)
        return g.subgraph([account_id]).copy()
    nodes = {account_id}
    frontier = {account_id}
    g_u = g.to_undirected()
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            nxt.update(g_u.neighbors(n))
        frontier = nxt - nodes
        nodes |= frontier
    return g.subgraph(nodes).copy()
