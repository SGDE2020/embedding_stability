from pathlib import Path
from lib.tools import parse_graph, remove_edges_randomly, sample_negative_edges, sample_positive_edges
import networkx as nx
from pickle import dump, load
import random
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("graph", type=str)
parser.add_argument("--use-already-shrunken-graph", action="store_true")
parser.add_argument("--relabel-nodes", action="store_true")
args = parser.parse_args()

# just to have a fixed seed
np.random.seed(0)
random.seed(0)

graph_name = args.graph
_, graph = parse_graph(f'./graphs/{graph_name}/', largest_cc=True)

if args.relabel_nodes:
    mapping = dict(zip(graph.nodes(), range(graph.number_of_nodes())))
    with open(f'./graphs/{graph_name}/{graph_name}_reduced_edges.pickle', 'rb') as file:
        true_edges = load(file)
    with open(f'./graphs/{graph_name}/{graph_name}_reduced_nonedges.pickle', 'rb') as file:
        false_edges = load(file)
    with open(f'./graphs/{graph_name}/{graph_name}_reduced_nonedges_train.pickle', 'rb') as file:
        false_edges_train = load(file)
    with open(f'./graphs/{graph_name}/{graph_name}_reduced_edges_train.pickle', 'rb') as file:
        true_edges_train = load(file)
        map_edge = lambda e: (mapping[e[0]], mapping[e[1]])
    true_edges = set(map(map_edge, true_edges))
    false_edges = set(map(map_edge, false_edges))
    false_edges_train = set(map(map_edge, false_edges_train))
    true_edges_train = set(map(map_edge, true_edges_train))
else:

    if args.use_already_shrunken_graph:
        _, small_graph = parse_graph(f'./graphs/{graph_name}/{graph_name}_reduced.edgelist')
        diff_graph = nx.difference(graph,small_graph)
        true_edges = diff_graph.edges()
    else:
        true_edges = remove_edges_randomly(graph, frac=0.1, keep_connected=True)
        nx.write_edgelist(graph, f'./graphs/{graph_name}/{graph_name}_reduced.edgelist')

    false_edges_train = sample_negative_edges(graph, frac=0.9)
    false_edges = sample_negative_edges(graph, frac=0.1, exclude=false_edges_train)

    true_edges_train = sample_positive_edges(graph, frac=1)

with open(f'./graphs/{graph_name}/{graph_name}_reduced_nonedges.pickle', 'wb') as file:
    dump(false_edges, file)
with open(f'./graphs/{graph_name}/{graph_name}_reduced_nonedges_train.pickle', 'wb') as file:
    dump(false_edges_train, file)
with open(f'./graphs/{graph_name}/{graph_name}_reduced_edges_train.pickle', 'wb') as file:
    dump(true_edges_train, file)
with open(f'./graphs/{graph_name}/{graph_name}_reduced_edges.pickle', 'wb') as file:
    dump(true_edges, file)


