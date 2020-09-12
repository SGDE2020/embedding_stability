import argparse
from collections import defaultdict
import logging
import os

import networkx as nx
import numpy as np
import pandas as pd

from lib.tools import parse_graph
from lib.tools.comparison import Comparison


def neighbor_variance(graphdir, k, embedding_dir, embedding_list, dataset, nodes_dict=None, seed=None):
    """ 
    Compute similarity from nodes to specific kind of neighbors: 1-hop, 2-hop neighbors and 
    distant nodes (more than 2-Hops distance). The nodes will be sampled from the graph. Since some nodes
    may not have all required kinds of neighbors, the actual size of the experiment might be lower than expected.

    Args:
        graphdir: str, path to graph file (e.g. edgelist)
        k: int, number of nodes for which the similarities will be computed
        embedding_dir: str, path to directory of saved embeddings
        embedding_list: list, list of all embedding names that will be considered
        dataset: str, graph identifier
        nodes_dict: dict (optional), a dict of 3-tuples or lists of node ids. The tuples are ordered as 1-hop,
                    2-hop, distant. The keys are the origin node corresponding to the tuple. No nodes will be sampled.
        seed: int, sets the random seed for numpy

    Returns:
        agg_results: numpy array of size k x 3. Every row correponds to one sampled node and its mean 
            similarities to its pair nodes.
        DataFrame: pandas DataFrame that hold information about the dataset, algorithm, node and similarity 
            measure to a neighbor node. (neighbor_type: 0 -> 1-neighbor, 1 -> 2-neighbor, 2 -> distant node)
        di: dict with sampled nodes as keys and corresponding lists of 1-neighbor, 2-neighbor and distant node. 
            An entry is None if there could not be found a node with the correct characteristics.
    """

    if seed is not None:
        np.random.seed(seed)

    if nodes_dict is None:
        graph_name, graph = parse_graph(graphdir)

        # Sample k nodes if not specified by arguments
        vertices = np.array(list(graph))
        np.random.shuffle(vertices)
        nodes = vertices[:min(k, graph.number_of_nodes())].copy()
        
        # Find 1-hop, 2-hop and distant nodes for every sampled node
        di = defaultdict(list)
        for node in nodes:
            # 1-hop: sample neighbor that is not the node itself
            one_neighbors = list(graph.neighbors(node))
            if node in one_neighbors:
                one_neighbors.remove(node)
            # If there is no neighbor, just skip the node
            if not one_neighbors:
                di[node].extend([None, None, None])
                continue
            else:
                di[node].append(np.random.choice(one_neighbors))
            # 2-hop: sample neighbor; from there another node, that has no edge to the origin
            one_neighbor = np.random.choice(one_neighbors)
            one_neighbors.append(node)
            two_n = None
            for two_n_candidate in graph.neighbors(one_neighbor):
                if two_n_candidate not in one_neighbors:
                    two_n = two_n_candidate
                    break
            di[node].append(two_n)
            # distant node: sample random node, compute shortest path, if length is more than 2 accept the sample
            np.random.shuffle(vertices)
            distant_node = None
            for distant_node_candidate in vertices:
                try:
                    if nx.shortest_path_length(graph, distant_node_candidate, node) > 2:
                        distant_node = distant_node_candidate
                        break
                except nx.NetworkXNoPath:
                    distant_node = distant_node_candidate
                    break
            di[node].append(distant_node)
            assert len(di[node]) == 3, f"{node}: {di[node]}"
    else:
        di = nodes_dict
        nodes = np.asarray(list(nodes_dict.keys()))
    
    # Aggregate all node for comparison
    cossim_nodes = []
    for origin_node, node_list in di.items():
        cossim_nodes.append(origin_node)
        cossim_nodes.extend(node_list)
    cossim_nodes = sorted(list(set([node for node in cossim_nodes if node is not None])))
    array_map = dict([(node, i) for i, node in enumerate(cossim_nodes)])

    # Start Comparison.cosine_similarity with selected nodes
    comparison = Comparison(embedding_dir, embedding_list)
    comp_result = comparison.cosine_similarity(cossim_nodes)["sims"]

    # Analyse the result
    # Create a matrix, with columns 1-hop, 2-hop, distant neighbor;
    # rows are nodes from the sample; in every cell the mean is noted
    agg_results = np.empty((len(nodes), 3))
    # Construct a DataFrame that holds the detailed results
    # DataFrame has columns: node, neighbor_type, similarity, algorithm, dataset
    df_dict = defaultdict(list)
    algorithm = embedding_list[0].split("_")[0]
    
    for index, node in enumerate(nodes):
        nr = {0: [], 1: [], 2: []}
        for i, v in enumerate(di[node]):
            if v is None:
                nr[i] = [np.nan]
                df_dict["node"].append(node)
                df_dict["neighbor_type"].append(i)
                df_dict["algorithm"].append(algorithm)
                df_dict["dataset"].append(dataset)
                df_dict["similarity"].append(np.nan)
                continue
            for arr in comp_result.values():
                nr[i].append(arr[array_map[node], array_map[v]])
                df_dict["node"].append(node)
                df_dict["neighbor_type"].append(i)
                df_dict["algorithm"].append(algorithm)
                df_dict["dataset"].append(dataset)
                df_dict["similarity"].append(arr[array_map[node], array_map[v]])
        agg_results[index, :] = [np.mean(nr[0]), np.mean(nr[1]), np.mean(nr[2])]

    # Remove rows with NaNs
    agg_results = agg_results[~np.isnan(agg_results).any(axis=1)]
    print(f"Removed {len(nodes) - len(agg_results)} nodes from the sample due to them not having correct neighbors.")
    return agg_results, pd.DataFrame(df_dict), di

def get_file_list(filter_mask, directory):
    if isinstance(filter_mask, str):
        print(filter_mask in os.listdir(directory)[-2])
        return sorted([f for f in os.listdir(directory) if filter_mask in f and f.endswith(".emb")])
    else:
        return sorted([f for f in os.listdir(directory) if
                    (len(filter_mask) == len(list(filter(lambda x: x in f, filter_mask))) and f.endswith(".emb"))])


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-a", "--algorithm")
    arg_parser.add_argument("-d", "--dataset")
    arg_parser.add_argument("-r", "--resultsdir")
    arg_parser.add_argument("-g", "--graphdir")
    arg_parser.add_argument("-k", type=int)
    arg_parser.add_argument("-s", "--seed", type=int)
    return arg_parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    embedding_list = get_file_list([args.algorithm, args.dataset], args.resultsdir)
    print(embedding_list)
    r, df, nb_dict = neighbor_variance(args.graphdir, args.k, args.resultsdir, embedding_list, 
                                        args.dataset, seed=args.seed)
    print("Means:\t", np.mean(r, axis=0))
    print("Stds:\t", np.std(r, axis=0))
    print("Cvs:\t", np.divide(np.std(r, axis=0), np.mean(r, axis=0)))
    print(df)