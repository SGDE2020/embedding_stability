"""Creates HOPE embeddings
"""
from time import time
from lib.tools.embedding import save_embedding
from lib.tools import parse_graph

from gem.embedding.hope import HOPE

import networkx as nx
import numpy as np
import random
import os
import argparse

def parse_args():
    """Parse arguments given to the script

    Returns:
        argparse.ArgumentParser.parse_args -- Dict like
            object containing the passed arguments
    """
    arg_parser = argparse.ArgumentParser(
        description="Runs the HOPE embedding algorithm on a given graph.")

    # Require arguments
    arg_parser.add_argument("dataset", type=str,
                            help="Path to the dataset that should be used")
    # Optional
    arg_parser.add_argument("--outputdir", type=str,
                            default=os.path.dirname(os.path.abspath(__file__)) + "/results/",
                            help="Path to output directory")

    # General options
    arg_parser.add_argument("--largest-cc", action="store_true",
                            help=("States whether only the largest connected component "
                                  "will be used, which might be useful when dealing with "
                                  "disconnected graphs"))
    arg_parser.add_argument("-n", "--num_embeddings", type=int, default=1,
                            help="Sets the number of embeddings that will be created")
    arg_parser.add_argument("-s", "--seed", type=int, help="Sets the random seed", default=None)

    # HOPE specific parameters
    arg_parser.add_argument("-d", "--dimensions", type=int, default=128, help="Sets the dimensionality of the embedding")
    arg_parser.add_argument("-b", "--beta", type=float, default=0.01, help="Sets the decay parameter in Katz similarity")

    return arg_parser.parse_args()

def main():
    """Main execution steps
    Reads arguments, fixes random seeding, executes the HOPE model
    and saves resulting embeddings.
    """
    args = parse_args()

    # Set random seed if specified
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
    # numpy seeds need to be in [0, 2^32-1]
    args.seed = [np.random.randint(4294967296 - 1) for i in range(args.num_embeddings)]


    # Parse graph data
    graph_name, graph = parse_graph(args.dataset, args.largest_cc)
    graph = graph.to_directed()

    # Compute embeddings
    model = HOPE(d=128, beta=args.beta)

    print("Num nodes: %d, num edges: %d" % (graph.number_of_nodes(), graph.number_of_edges()))
    times = []
    for i in range(args.num_embeddings):
        t1 = time()

        # Set the seed before learning
        np.random.seed(args.seed[i])
        random.seed(args.seed[i])

        Y, t = model.learn_embedding(graph=graph, edge_f=None, is_weighted=True, no_python=True)
        times.append(time() - t1)

        # save embedding
        file_path = (f"{args.outputdir}/hope_{graph_name}_"
                     f"{i:03d}.emb")
        print(f"Saving embedding to {file_path}")
        save_embedding(Y, file_path, {
            "algorithm": "hope",
            "dimension": args.dimensions,
            "beta": args.beta,
            "seed": args.seed[i],
            })

    print(model._method_name+"\n\tAverage training time: %f" % (sum(times)/len(times)))

if __name__ == "__main__":
    main()