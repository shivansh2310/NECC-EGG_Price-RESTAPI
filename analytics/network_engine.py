from __future__ import annotations

import networkx as nx
import pandas as pd

from analytics.correlation_engine import price_matrix


def correlation_network(prices: pd.DataFrame, threshold: float = 0.65) -> nx.Graph:
    """Build an undirected network from strong absolute return correlations."""
    returns = price_matrix(prices).pct_change(fill_method=None)
    corr = returns.corr(min_periods=30)
    graph = nx.Graph()
    graph.add_nodes_from(corr.columns)
    for index, market_a in enumerate(corr.columns):
        for market_b in corr.columns[index + 1 :]:
            value = corr.loc[market_a, market_b]
            if pd.notna(value) and abs(value) >= threshold:
                graph.add_edge(
                    market_a,
                    market_b,
                    correlation=float(value),
                    weight=float(abs(value)),
                    distance=float(1 - abs(value)),
                )
    return graph


def centrality_table(graph: nx.Graph) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["market", "degree", "betweenness", "eigenvector"])
    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, weight="distance")
    try:
        eigenvector = nx.eigenvector_centrality(graph, weight="weight", max_iter=1_000)
    except (nx.NetworkXException, nx.PowerIterationFailedConvergence):
        eigenvector = {node: 0.0 for node in graph}
    return (
        pd.DataFrame(
            {
                "market": list(graph.nodes),
                "degree": [degree[node] for node in graph],
                "betweenness": [betweenness[node] for node in graph],
                "eigenvector": [eigenvector[node] for node in graph],
            }
        )
        .sort_values(["eigenvector", "degree"], ascending=False)
        .reset_index(drop=True)
    )


def lead_lag_network(
    prices: pd.DataFrame, lag: int = 1, threshold: float = 0.35
) -> nx.DiGraph:
    """Connect A→B when A's return is correlated with B's future return."""
    returns = price_matrix(prices).pct_change(fill_method=None)
    graph = nx.DiGraph()
    graph.add_nodes_from(returns.columns)
    for market_a in returns:
        for market_b in returns:
            if market_a == market_b:
                continue
            value = returns[market_a].corr(returns[market_b].shift(-lag), min_periods=30)
            reverse = returns[market_b].corr(returns[market_a].shift(-lag), min_periods=30)
            if pd.notna(value) and abs(value) >= threshold and abs(value) > abs(reverse):
                graph.add_edge(
                    market_a,
                    market_b,
                    correlation=float(value),
                    weight=float(abs(value)),
                    distance=float(1 - abs(value)),
                )
    return graph
