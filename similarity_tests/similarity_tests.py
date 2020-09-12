""" Run this file to conduct similarity tests
Example usage: python3 similarity_tests.py --emb-dir {DIRECTORY OF EMBEDDINGS} -a {ALGORITHMS} -d {DATASETS}  -t {TESTS}
"""

import argparse
import os
import warnings
from itertools import product

import pandas as pd

from lib.tools.comparison import Comparison

TESTS = ["cossim", "knn", "jaccard", "ranks", "rns", "angdiv", "2ndcos", "orthtf", "cos", "lintf", "lincos"]
ALGORITHMS = ["line", "hope", "sdne", "graphsage", "node2vec"]
DATASETS = ["cora", "facebook", "blogcatalog", "protein", "wikipedia"]


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    # Test parameter
    parser.add_argument(
        "-a", "--algorithms", nargs="+", type=str,
        choices=ALGORITHMS, default=ALGORITHMS,
        help="Algorithms used in evaluation."
    )
    parser.add_argument(
        "-d", "--datasets", nargs="+", type=str,
        default=DATASETS,
        help="Datasets used in evaluation. Argument must be part of embedding file name."
    )
    parser.add_argument(
        "-t", "--tests", nargs="+", type=str,
        choices=TESTS, default=["cossim", "jaccard", "ranks", "rns", "angdiv"],
        help="List of tests which will be conducted."
    )
    parser.add_argument(
        "--num-nodes", type=int, default=-1,
        help="How many nodes will be used in the tests. By default, all nodes are considered."
    )
    parser.add_argument(
        "--knn-size", type=int, default=20,
        help="Size of neighborhood in knn tests."
    )

    # General options
    parser.add_argument(
        "--load-knn", type=bool, default=False,
        help="Whether to load an already computed knn matrix."
    )
    parser.add_argument(
        "--kload-size", type=int,
        help="Value of k of the knn file that is to be loaded. Has to be bigger or equal to knn-size."
    )
    parser.add_argument(
        "--emb-dir", type=str,
        required=True,
        help="Where to find the embeddings")
    parser.add_argument(
        "--nodeinfo-dir", type=str,
        required=True,
        help="Where to find the node info files"
    )
    parser.add_argument(
        "--results-dir", type=str,
        required=True,
        help=("Where to find save the results to. "
              "Requires a subdirectory called 'procrustes_matrices'")
    )
    parser.add_argument(
        "--num-processes", type=int, default=4,
        help="Number of processes in parallelization"
    )
    return parser.parse_args()


def get_file_list(filter_mask, directory):
    """ Computes a list of all .emb files that include all parts of the mask in their file name. """
    if isinstance(filter_mask, str):
        return sorted([f for f in os.listdir(directory) if filter_mask in f and f.endswith(".emb")])
    else:
        return sorted([f for f in os.listdir(directory) if
                       (len(filter_mask) == len(list(filter(lambda x: x in f, filter_mask))) and f.endswith(".emb"))])


def sample_nodes(df, num_samples):
    """ Return all nodes of a graph from its nodeinfo DataFrame
    Params:
        df: pandas.DataFrame, indexed by node ids of a graph
        num_samples: int, number of nodes to be sampled
    Returns:
        list, contains index of df
    Raises:
        NotImplementedError, if num_samples >= 0
    """
    ids = list(df.index)
    if num_samples < 0:
        return ids
    else:
        # Sample random nodes, should be reproducible via seed
        raise NotImplementedError


def run_tests(embedding_dir, algorithms=ALGORITHMS, datasets=DATASETS, tests=TESTS,
              num_nodes=-1, knn_size=20, load_knn=False, kload_size=None,
              nodeinfo_dir="./node_info", results_dir="./results/", num_processes=4):
    """
    Run specified similarity tests.
    Params:
        embedding_dir: str, path to directory where embeddings are stored
        algorithms: list of str, algorithms that will be considered in the tests
        datasets: list of str, datasets (graphs) that will be considered in the tests
        tests: list of str, tests that will be conducted
        num_nodes: int, number of nodes of a graph that are used in the tests
        knn_size: int, neighborhood size for knn test
        load_knn: bool, whether a stored knn matrix should be loaded
        nodeinfo_dir = str, path to directory where nodeinfo is stored (tables)
        results_dir = str, path to directory where results will be saved to
    """
    # For every (algorithm, dataset) pair, find their embedding file names and then conduct the tests
    for alg, dataset in product(algorithms, datasets):
        # Create list of file names
        fnames = (get_file_list(alg + "_" + dataset, embedding_dir))
        print(fnames)

        if len(fnames) <= 1:
            warnings.warn(f"Did not find any embeddings for algorithm {alg} and dataset {dataset}. "
                          "Continuing with next combination..")
            continue

        # Read nodeinfo files to sample nodes from the graph
        info = pd.read_csv(nodeinfo_dir + dataset + ".node_info", sep=" ", header=0, index_col=0)
        nodes = sample_nodes(info, num_nodes)

        if load_knn and kload_size is None:
            kload_size = knn_size

        # Start tests
        comp = Comparison(emb_dir=embedding_dir, embeddings=fnames)
        if "cossim" in tests:
            print("Aligned cosine similarity")
            comp.cossim_analysis(nodes=nodes, save_path=results_dir)
        if "knn" in tests:
            print("Executing k nearest neighbor overlap")
            comp.k_nearest_neighbors(
                nodes=nodes, append=False, k=knn_size,
                load=load_knn, kload_size=kload_size, save=True, save_path=results_dir, num_processes=num_processes
            )
        if "jaccard" in tests:
            print("Executing jaccard score")
            comp.jaccard_similarity(
                nodes=nodes, append=False, k=knn_size,
                load=load_knn, kload_size=kload_size, save=True, save_path=results_dir, num_processes=num_processes
            )
        if "ranks" in tests:
            print("Executing rank invariance score")
            comp.ranking_invariance(
                nodes=nodes, append=False, k=knn_size,
                load=load_knn, kload_size=kload_size, save=True, save_path=results_dir, num_processes=num_processes
            )
        if "2ndcos" in tests:
            print("Executing second order cosine similarity")
            comp.second_order_cosine_similarity(
                nodes=nodes, append=False, k=knn_size,
                save=True, save_path=results_dir, num_processes=num_processes
            )
        if "angdiv" in tests:
            print("Executing second order angle divergence")
            comp.knn_angle_divergence(
                nodes=nodes, append=False, k=knn_size,
                load=load_knn, kload_size=kload_size, save=True, save_path=results_dir, num_processes=num_processes
            )
        if "rns" in tests:
            print("Executing ranked neighborhood stability")
            comp.ranked_neighborhood_stability(k=knn_size, save=True, save_path=results_dir)
        if "cos" in tests:
            print("Executing cosine similarity")
            comp.cosine_similarity(
                nodes=nodes, append=False, num_samples=100, save=True,
                save_path=results_dir
            )
        if "orthtf" in tests:
            print("Executing Procrustes transformation")
            comp.procrustes_transformation(save_path=results_dir)
        if "lintf" in tests:
            print("Executing Procrustes with linear shift transformation")
            comp.linproc_transformation(save_path=results_dir)
        if "lincos" in tests:
            print("Executing Procrustes with linear shift analysis")
            comp.linproc_analysis(nodes=nodes, save_path=results_dir)


if __name__ == "__main__":
    args = parse_args()
    run_tests(
        args.emb_dir, args.algorithms, args.datasets, args.tests,
        args.num_nodes, args.knn_size, args.load_knn,
        args.kload_size, args.nodeinfo_dir, args.results_dir, args.num_processes
    )
