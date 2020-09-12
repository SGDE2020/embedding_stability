""" Creates LINE embeddings

    The implementation is based on https://github.com/tangjianpku/LINE.
"""
import os
import subprocess
import multiprocessing
import networkx as nx
import argparse
import numpy as np
from time import time
from lib.tools.nx_to_edgelist import nx_to_weighted_edgelist
from lib.tools import parse_graph
from lib.tools.embedding import read_embedding, save_embedding, create_param_lines, prepend_param_lines


GRAPH_SOURCE_SNAP = "snap"
GRAPH_SOURCE_KONECT = "konect"
GRAPH_SOURCE_ASU = "asu"
GRAPH_SOURCE_SYNTH ="synth"
LINE_EXECUTABLES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../", "lib/line_embedding/COMPILED_LINE/linux"))

def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Runs the LINE embedding algorithm on a given graph.")

    # Require arguments
    arg_parser.add_argument("dataset", type=str,
                            help="Path to the dataset that should be used")
    # Optional
    arg_parser.add_argument("--outputdir", type=str,
                            default=os.path.dirname(os.path.abspath(__file__)) + "/results/",
                            help="Path to output directory")

    # General options
    arg_parser.add_argument("--executables", type=str, default=LINE_EXECUTABLES, help="Path to LINE executables")
    arg_parser.add_argument("--largest-cc", action="store_true",
                            help=("States whether only the largest connected component "
                                  "will be used, which might be useful when dealing with "
                                  "disconnected graphs"))
    arg_parser.add_argument("-n", "--num_embeddings", type=int, default=1,
                            help="Sets the number of embeddings that will be created")
    arg_parser.add_argument("-t", "--num-threads", type=int, default=multiprocessing.cpu_count(),
                            help="Sets the number of threads LINE will use")
    arg_parser.add_argument("--seed", type=int, help="Set the random seed. "
                                                    #"For every of num_embeddings runs, a new seed for the"
                                                    #" random initilization of the embedding is set. The new seed is sampled from a distribution seeded by the given parameter."
                                                    #" However, the embeddings will only be reproducible when num_threads=1 due to the optimization method."
                                                    ,
                            default=None)

    # LINE specific parameters
    arg_parser.add_argument("-dim", "--dimensions", type=int, default=128, help="Sets the dimensionality of the embedding")
    arg_parser.add_argument("-neg", "--negative-samples", type=int, default=5, help="Sets the number of negative samples")
    arg_parser.add_argument("-s", "--samples", type=int, default=None, help="The total number of training samples in million")
    arg_parser.add_argument("-thresh", "--threshold", type=int, default=None,
                            help="The number of second order neighbors considered for densification")
    arg_parser.add_argument("--densify", type=bool, default=True, help="Whether the embedding will be learned on the densified graph")
    arg_parser.add_argument("--normalize", type=bool, default=True, help="Whether the embedding will be normalized")

    return arg_parser.parse_args()

def main():
    # Load arguments
    args = parse_args()

    # Parse graph data
    graph = None
    graph_name = ""
    print(f"Parsing graph dataset from {args.dataset}")#, created by {args.source}")
    graph_name, graph = parse_graph(args.dataset, args.largest_cc)
    print(f"{graph_name} has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    # Create an undirected copy of the graph for the first order embedding.
    # Directed graphs will change whereas an undirected graph will stay the same.
    print(f"Graph has {graph.number_of_edges()} edges")
    if type(graph) == nx.Graph:
        print("Graph is undirected")
        graph = graph.to_directed()
        undirected_graph = graph
    else:
        print("Graph is directed")
        undirected_graph = graph.to_undirected()
        undirected_graph = undirected_graph.to_directed()

    node_count = len(graph)

    # Densify graph
    if args.threshold is None:
        # calculate the threshold from graph characteristics
        print("Calculating threshold...")
        avg_deg = 2*graph.number_of_edges()/graph.number_of_nodes()
        args.threshold = max(0, int(avg_deg))
        print("nodes", node_count)
        print("threshold", args.threshold)
    path = None
    if os.path.isdir(args.dataset) is False:
        path = os.path.split(args.dataset)[0]
    else:
        path = args.dataset
    print("Densifying graph...")
    if args.densify is True:
        nx_to_weighted_edgelist(graph, f"{path}/{graph_name}.Edgelist")
        subprocess.call(f"{args.executables}/reconstruct -train {path}/{graph_name}.Edgelist "
                        f"-output {path}/{graph_name}.denseEdgelist -depth 2 -threshold {args.threshold}",
                        shell=True)
        nx_to_weighted_edgelist(undirected_graph, f"{path}/{graph_name}_undir.Edgelist")
        subprocess.call(f"{args.executables}/reconstruct -train {path}/{graph_name}_undir.Edgelist "
                        f"-output {path}/{graph_name}_undir.denseEdgelist -depth 2 -threshold {args.threshold}",
                        shell=True)

    # Handle seed
    if args.seed is None:
        # srand(0) --> random seed for every iteration
        args.seed = [0 for i in range(args.num_embeddings)]
    else:
        np.random.seed(args.seed)
        args.seed = [np.random.randint(4294967296 - 1) for i in range(args.num_embeddings)]

    # Learn embeddings
    if args.samples is None:
        args.samples = max(node_count // 1000, 1)
    timings =[]
    for i in range(args.num_embeddings):
        t1 = time()
        # 1st order
        subprocess.call(f"{args.executables}/line -train {path}/{graph_name}_undir.denseEdgelist "
            f"-output {args.outputdir}/line_1st_order_{graph_name}_{i:03d} -binary 1 -size {args.dimensions//2} "
            f"-order 1 -negative {args.negative_samples} -samples {args.samples} -threads {args.num_threads}"
            f"-srand {args.seed[i]}",
            shell=True)
        file_path_1 = f"{args.outputdir}/line_1st_order_norm_{graph_name}_{i:03d}.emb"
        subprocess.call(f"{args.executables}/normalize -input {args.outputdir}/line_1st_order_{graph_name}_{i:03d} "
            f"-output {file_path_1} -binary 0",
            shell=True)

        # 2nd order
        subprocess.call(f"{args.executables}/line -train {path}/{graph_name}.denseEdgelist "
            f"-output {args.outputdir}/line_2nd_order_{graph_name}_{i:03d} -binary 1 -size {args.dimensions//2} "
            f"-order 2 -negative {args.negative_samples} -samples {args.samples} -threads {args.num_threads}"
            f"-srand {args.seed[i]}",
            shell=True)
        file_path_2 = f"{args.outputdir}/line_2nd_order_norm_{graph_name}_{i:03d}.emb"
        subprocess.call(f"{args.executables}/normalize -input {args.outputdir}/line_2nd_order_{graph_name}_{i:03d} "
            f"-output {file_path_2} -binary 0",
            shell=True)

        # add meta information
        param_lines = create_param_lines({
            "node_count": node_count,
            "embedding_dimension": f"{args.dimensions//2}"
        })
        prepend_param_lines(file_path_1, param_lines)
        prepend_param_lines(file_path_2, param_lines)

        # Concatenate embeddings
        print("Concatenating embeddings...")
        with open(file_path_1, "r") as f:
            first_order = read_embedding(f)
        with open(file_path_2, "r") as f:
            second_order = read_embedding(f)
        concat = np.concatenate((first_order, second_order), axis=1)
        save_embedding(concat, f"{args.outputdir}/line_{graph_name}_{i:03d}.emb", {
            "algorithm": "line",
            "order": "first+second",
            "negative_samples": args.negative_samples,
            "samples": args.samples,
            "threshold": args.threshold,
            "densify": args.densify,
            "node_count": node_count,
            "embedding_dimension": args.dimensions,
            "threads": args.num_threads,
            "seed": args.seed[i],
            "comment": "1st-order-and-2nd-order-get-the-same-dimensionality"
        })
        timings.append(time() - t1)

        # Clean up
        subprocess.call(f'rm {args.outputdir}/line_1st_order_{graph_name}_{i:03d}', shell=True)
        subprocess.call(f'rm {args.outputdir}/line_2nd_order_{graph_name}_{i:03d}', shell=True)
        subprocess.call(f'rm {args.outputdir}/line_1st_order_norm_{graph_name}_{i:03d}.emb', shell=True)
        subprocess.call(f'rm {args.outputdir}/line_2nd_order_norm_{graph_name}_{i:03d}.emb', shell=True)

    # Clean up
    subprocess.call(f"rm {path}/{graph_name}.Edgelist", shell=True)
    subprocess.call(f"rm {path}/{graph_name}_undir.Edgelist", shell=True)

    print(f"\nDone!\nAverage training time: {sum(timings)/len(timings)}")

if __name__ == "__main__":
    main()