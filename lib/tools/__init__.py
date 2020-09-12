from itertools import combinations_with_replacement
from random import choices, sample

import networkx as nx

from .parse_graph import parse_graph

def number_random_edges(graph: nx.Graph, *, frac=None, number=None):
    if frac is None and number is None:
        raise ValueError("Either a fraction of edges or an absolute number of edges to be removed "
                         "must be provided. Set either `frac` or `number`.")

    if frac is not None and number is not None:
        raise ValueError("Can only specify either a fraction of edges or an absolute number of "
                         "edges to be removed, but not both.")

    return int(len(graph.edges) * frac) if frac else number

def sample_positive_edges(graph: nx.Graph, *, frac=None, number=None):
    number = number_random_edges(graph, frac=frac, number=number)
    return sample(graph.edges, number)

def remove_edges_randomly(graph: nx.Graph, *, frac=None, number=None, keep_connected=False):
    """
    Removes edges randomly from the given graph. You must either give
    `frac` or `number` to specify how many edges should be removed.

    The parameter `keep_connected` can be used to ensure that all nodes remain
    connected. Some embedding algorithms (DAOR) lead to subsequent problems in
    later tasks otherwise.

    A set of all edges that were removed is returned. The graph is altered in
    place.
    """
    number = number_random_edges(graph, frac=frac, number=number)

    connected = nx.is_weakly_connected if graph.is_directed() else nx.is_connected
    if keep_connected and not connected(graph):
        raise RuntimeError("The graph is already not connected before removing any edges.")

    removed_edges = set()
    infeasible_edges = set()
    while len(removed_edges) < number:
        to_remove = sample(graph.edges - infeasible_edges, number - len(removed_edges))

        for edge in to_remove:
            edge_data = graph[edge[0]][edge[1]]
            graph.remove_edge(*edge)
            if keep_connected and not connected(graph):
                # The removed edge made the graph disconnected, add the edge
                # again and re-sample a new edge as replacement later.
                graph.add_edge(*edge, **edge_data)
                infeasible_edges.add(edge)
            else:
                removed_edges.add(edge)

    return removed_edges

def sample_negative_edges(graph: nx.Graph, *, frac=None, number=None, exclude=None):
    number = number_random_edges(graph, frac=frac, number=number)
    if exclude is None:
        exclude = set()

    negative_edges = set()
    while len(negative_edges) < number:
        us = choices(list(graph.nodes), k=number - len(negative_edges))
        vs = choices(list(graph.nodes), k=number - len(negative_edges))

        for u, v in zip(us, vs):
            if u == v:
                # we don't want to generate self-loops
                continue
            if v < u and not graph.is_directed():
                # for undirected graphs don't consider edges in both directions
                continue
            if graph.has_edge(u, v):
                continue
            if (u, v) in exclude:
                continue
            negative_edges.add((u, v))

    return negative_edges
