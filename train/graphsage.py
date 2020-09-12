"""
This script calls the official GraphSAGE implementation as can be found at
https://github.com/williamleif/GraphSAGE

We made slight modifications to make the code work on newer versions of
networkx, but refrained from refactoring the code further, e.g. such that no
temporary files are used as interface to GraphSAGE.

We originally planned to use StellarGraph's GraphSAGE implementation, but
StellarGraph does not support "identity features" when no node attributes are
available. See https://github.com/stellargraph/stellargraph/issues/69
"""
import argparse
import os
import json
import subprocess
import tempfile
import random

import networkx as nx
import numpy as np

from lib.graphsage_embedding.graphsage.utils import run_random_walks
from lib.tools.embedding import save_embedding
from lib.tools import parse_graph


def generate(graph, seed):
    for node in graph:
        # We use GraphSAGE in unsupervised mode, thus the test and validation
        # sets are empty. GraphSAGE requires us to set them explicitly anyway.
        graph.nodes[node]["val"] = False
        graph.nodes[node]["test"] = False

        # We don't need the node labels to generate node embeddings. For the
        # wikipedia dataset, they cannot be serialized into JSON, which makes
        # `nx.readwrite.json_graph.node_link_data` throw an exception.
        # Thus we simply remove them from the graph.
        if "class" in graph.nodes[node]:
            del graph.nodes[node]["class"]

    data = nx.readwrite.json_graph.node_link_data(graph)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # "graph-G.json" contains the input graph in node-link format
        with open(tmpdirname + "/graph-G.json", "w") as file:
            json.dump(data, file)

        # GraphSAGE expects an extra file with mappings from each node to
        # consecutive integers. This should already be the case, but just to
        # make sure, we enumerate through the graph instead of just using an
        # identity mapping of node ids.
        with open(tmpdirname + "/graph-id_map.json", "w") as file:
            id_map = dict((node, i) for i, node in enumerate(graph))
            json.dump(id_map, file)

        # Nodes in a graph can be mapped to different classes. We don't make
        # use of that feature and assign all nodes to the same class `1`.
        with open(tmpdirname + "/graph-class_map.json", "w") as file:
            class_map = {node: 1 for node in graph}
            json.dump(class_map, file)

        # Use GraphSAGE's random walk implementation (unbiased) to generate
        # random walk co-occurrences.
        pairs = run_random_walks(graph, graph.nodes)
        with open(tmpdirname + "/graph-walks.txt", "w") as fp:
            fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))

        exec_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../", "lib/graphsage_embedding"))

        subprocess.run([
            "python3", "-m", "graphsage.unsupervised_train",
            "--train_prefix", tmpdirname + "/graph",
            "--model", "graphsage_mean",
            "--identity_dim", "128",
            "--base_log_dir", tmpdirname,
            "--dim_1", "64",
            "--dim_2", "64",
            "--seed", str(seed)
            ],
            cwd=exec_dir
        )

        # GraphSAGE stores the embedding under a filename generated from the
        # parameters in the directory with the input files.
        # We load the embedding (numpy array) from their and use our own saving
        # methods for compatibility with other embedding algorithms.
        emb_file = tmpdirname + "/unsup-" + tmpdirname.split("/")[-1] + ("/graphsage_mean_small_{0:0.6f}/val.npy").format(0.00001)
        return np.load(emb_file)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Runs the unsupervised GraphSAGE representation learning algorithm on the given graph.")
    arg_parser.add_argument("dataset", nargs="+", type=str, help="Path to the dataset(s) for which embeddings should be generated.")
    arg_parser.add_argument("-largest-cc", action="store_true", help="States whether only the largest (weakly) connected component will be used, which might be useful when dealing with disconnected graphs.")
    arg_parser.add_argument("-num-embeddings", type=int, default=1, help="Number of embeddings that should be generated for each graph.")
    arg_parser.add_argument("--seed", type=int, help="Sets the random seed", default=None)

    args = arg_parser.parse_args()

    # Set inital seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Get a list of seeds for 'num_embeddings' many embeddings
    args.seed = [np.random.randint(4294967296 - 1) for i in range(args.num_embeddings)]

    for dataset in args.dataset:
        graph_name, graph = parse_graph(dataset, args.largest_cc)

        # Unfortunatly we need string labels for GraphSAGE. This blows up our
        # memory requirements to store the input graphs.
        nx.relabel_nodes(graph, {n: str(n) for n in graph}, copy=False)

        for i in range(args.num_embeddings):
            print("Generating embedding for round", i)
            embedding = generate(graph, args.seed[i])
            file_name = os.path.dirname(os.path.abspath(__file__)) + ("/results/graphsage_{0}_{1}.emb").format(graph_name, i)
            print("Save embedding at", file_name)
            save_embedding(embedding, file_name, {"algorithm": "graphsage", "seed": args.seed[i]})
