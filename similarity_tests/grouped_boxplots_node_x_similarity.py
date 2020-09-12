"""Creates a grouped boxplot graph with a selection of nodes on the x-axis and their
respective comparison results over all embeddings of a certain algorithm on the y-axis.
Multiple algorithms can be chosen for each node,
resulting in a group of boxplots for each node or node aggregation.
"""
import argparse
from lib.tools.node_info import parse_nodeinfo
from lib.tools.comparison import Comparison
from os import listdir
from os.path import isfile, join
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

COMPARISON_MODE_MANUAL = "manual"
COMPARISON_MODE_KNN = "knn"
COMPARISON_MODE_JACCARD = "jaccard"
COMPARISON_MODE_SO_CS = "so_cos_sim"
COMPARISON_MODE_PROC = "procrustes"

SELECTION_METHOD_SAMPLING = "sampling"
SELECTION_METHOD_MANUAL = "manual"
SELECTION_METHOD_IN_DEG = "in_deg"
SELECTION_METHOD_OUT_DEG = "out_deg"
SELECTION_METHOD_CORENESS = "coreness"
SELECTION_METHOD_PAGE_RANK = "page_rank"

def parse_args():
    """Parses the arguments given to the script in case of direct

    Returns:
        ArgumentParser.parse_args() -- The parsed arguments
    """
    arg_parser = argparse.ArgumentParser(
        description=("Creates a grouped boxplot graph, where the x-axis "
                     "holds a selection of nodes, the y-axis describe their "
                     "stability over multiple embeddings and one boxplot for each algorithm")
    )

    arg_parser.add_argument("output", type=str,
                            help="Specifies output destination")
    arg_parser.add_argument("comparison_mode", type=str,
                            choices=[
                                COMPARISON_MODE_MANUAL,
                                COMPARISON_MODE_KNN,
                                COMPARISON_MODE_JACCARD,
                                COMPARISON_MODE_SO_CS,
                                COMPARISON_MODE_PROC],
                            help=("States whether to use already created comparison results or "
                                  "create them automatically. Please see description for input "
                                  "for detailed information on the expected input format!"
                                  f"{COMPARISON_MODE_MANUAL}: The first file in inputs has to be "
                                  "a .npy file containing a list of the nodes used for comparison. "
                                  "Afterwards the result .npy-files should follow."
                                  "others: specified comparison method will be used to compare the "
                                  "embeddings given as input. For each algorithm a folder with its "
                                  "name has to be given in input, containing all the respective "
                                  "embedding files. Comparisons will be executed for each of these "
                                  "algorithms."))
    arg_parser.add_argument("inputs", type=str, nargs="+",
                            help=(f"If comparison_mode is set to {COMPARISON_MODE_MANUAL} the "
                                  "first file should contain the .npy-file containing the nodes "
                                  "used for the comparison (which is usually stored with the "
                                  "other files). Afterwards the result .npy-files of all "
                                  "executed comparisons should follow. It is assumed that the "
                                  "comparisons were executed using the same sample set of "
                                  "nodes! For other modes, folders should be given here with "
                                  "the containing the embeddings created by the different "
                                  "algorithms (e.g. sdne/ hope/ line2/)."))
    arg_parser.add_argument("-s", "--selection-method", type=str,
                            default=SELECTION_METHOD_MANUAL,
                            choices=[
                                SELECTION_METHOD_SAMPLING,
                                SELECTION_METHOD_MANUAL,
                                SELECTION_METHOD_IN_DEG,
                                SELECTION_METHOD_OUT_DEG,
                                SELECTION_METHOD_CORENESS,
                                SELECTION_METHOD_PAGE_RANK],
                            help=("Sets the method which will be used to select a subset of nodes."
                                  f"{SELECTION_METHOD_MANUAL}: Use all nodes"
                                  f"{SELECTION_METHOD_SAMPLING}: Samples --sample-size nodes"
                                  "others: Use nodes with highest, lowest "
                                  "and median node info attribute chosen. Node information will "
                                  "be gathered from --node-info file. "
                                  "Please notice that this will only select the extrema from "
                                  "the set given with nodes, which does not necessarily include "
                                  "the global extrema of the entire graph! If you want these "
                                  "overall nodes to be chosen, make sure to add them to nodes "
                                  "(and to the comparison results)."))
    arg_parser.add_argument("-S", "--sample-size", type=int, default=10,
                            help=("If --selection-method is set to "
                                  f"{SELECTION_METHOD_SAMPLING}, this specifies the sampling size"))
    arg_parser.add_argument("-A", "--aggregation-size", type=int, default=1,
                            help=("If --selection-method is not set to a node_info attribute "
                                  f"and this is set to a positive value, it specifies how many "
                                  "extrema should be considered. The maximum value is naturally "
                                  "defined by --sample-size/3."
                                  "For --aggregation-size = 5, the 5 nodes with lowest/centered/"
                                  "highest page_rank will be considered instead of a single one"))
    arg_parser.add_argument("-N", "--node-info", type=str, default=None,
                            help=("If --selection-method is set to "
                                  f"{SELECTION_METHOD_PAGE_RANK}, this points to the node info "
                                  "file of the graph"))
    arg_parser.add_argument("-k", type=int, default=100,
                            help=f"If comparison_mode is set to {COMPARISON_MODE_KNN}, this sets k")
    arg_parser.add_argument("--comparison-savepath", type=str, default=None,
                            help=(f"If comparison_mode is not set to {COMPARISON_MODE_MANUAL}, "
                                  "this sets the path to were all comparison files will be saved "
                                  "to. If left empty, nothing will be saved."))

    return arg_parser.parse_args()

def read_comparison_data_from_file(file_path):
    """Reads the results of a single comparison run

    Arguments:
        file_path {str} -- Path to the results file

    Returns:
        (str, pandas.DataFrame) -- Tuple of embedding algorithm used and DataFrame where each row
        corresponds to a compared node and the columns to the pair of embeddings
    """
    results = np.load(file_path)
    algorithm = file_path.split("/")[-1:][0].split("_")[0]
    return algorithm, results

def parse_comparison_data(data, all_nodes, node_ids=None):
    """Restructures comparison data to fit requirements for create_grouped_boxplots

    Arguments:
        data {dict} -- Dict with algorithms as keys and lists of lists or numpy.ndarrays as values
        List i corresponds to node[i] and its values are the comparison
        results of the embedding pairs
        node_ids {int[][]} -- List of lists of nodes that should be considered.
        If None, all available nodes will be used. Each sublist corresponds to a aggregation group.

    Returns:
        {pandas.DataFrame} -- DataFrame where each row corresponds to a single comparison result.
        The columns are:
            node_id: Id the node that was compared within all_nodes
            node: The node that was compared (actual node)
            algorithm: The embedding algorithm
            group: If nodes was aggregated, this holds the respective group
            pair_id: Id of the embedding pair that was compared
            comparison: The comparison metric result (e.g. overlap for KNN)
    """
    res = []

    # Pick all available indices from all_nodes
    if node_ids is None:
        node_ids = [range(len(all_nodes))]

    # Using DataFrame.melt might be more efficient
    for group_index, node_group in enumerate(node_ids):
        for node_id in node_group:
            for algorithm, comparison_results in data.items():
                for result_id, result in enumerate(comparison_results.T[node_id]):
                    res.append([node_id, all_nodes[node_id], algorithm,
                                group_index, result_id, result])

    data_frame = pd.DataFrame(data=res,
                              columns=
                              ["node_id", "node", "algorithm",
                               "group", "pair_id", "comparison"])
    return data_frame

def create_grouped_boxplots(data, file_path, x_key="node"):
    """Creates the grouped boxplot given the preprocessed dataframe

    Arguments:
        data {pandas.DataFrame} -- DataFrame as created by parse_comparison_data (long-format)
        Example of required values and format:
            #  algorithm  | node | (group) | comparison
            #      sdne   |   0  |   1     | 0.7
            #      sdne   |   0  |   1     | 0.65
            #      sdne   |   1  |   1     | 0.68
            #      sdne   |   2  |   1     | 0.65
            #      sdne   |   1  |   2     | 0.64
            #      line   |   0  |   2     | 0.64
        Here, the first two rows correspond to two different comparison results (embedding pairs).
        Aggregation will take place based on the groups column, if x_key is set to 'group'.

        file_path {str} -- Output file path

    Keyword Arguments:
        x_key {str} -- Key for the x-axis values (for aggregation pick "group") (default: {"node"})
    """
    figure = plt.figure()
    axis = sns.boxplot(x=x_key, y="comparison", hue="algorithm", data=data)
    figure.add_axes(axis)
    figure.savefig(file_path)

def choose_nodes_by_node_info(all_nodes, node_info, column, aggregation_size=1):
    """Selects a subsets of nodes based on a given node_info attribute

    Arguments:
        all_nodes {int[]} -- All available nodes as taken from a nodes.py file.
        These should contain the actual nodes (e.g. [256, 22, 9]) which were sampled
        during the comparison analysis

        node_info {pandas.DataFrame} -- DataFrame of node info as given by code.tools.node_info
        for given graph

        column {str} -- Column name to which the node_info should be filtered for (available
        options can be taken from parse_nodeinfo, e.g. page_rank, in_deg, ...)

    Keyword Arguments:
        aggregation_size {int} -- Specifies whether to aggregate x lowest/middle/highest
        page_rank nodes instead of the single "most extreme" one (default: {1})

    Returns:
        [int[][]] -- List of lists with length three. Each sublists represents a group. For a given
        aggregation_size a, the first list contains the a nodes of all_nodes with lowest page_rank,
        the second one the middle a and the last one the highest a page_rank nodes respectively
    """
    # Simply sort all nodes by their page rank
    frame = node_info[node_info.index.isin(all_nodes)].sort_values(by=column)

    # Pick nodes at beggining, center and end
    mid = int(len(frame)/2)
    df_low = frame[:aggregation_size]
    df_med = frame[mid-int(aggregation_size/2):mid+1+int(aggregation_size/2)]
    df_high = frame[-aggregation_size:]

    # Reverse lookup their IDs within all_nodes (as required by comparison results structure)
    nodes = [
        [np.where(all_nodes == node)[0][0] for node in list(df_low.index)],
        [np.where(all_nodes == node)[0][0] for node in list(df_med.index)],
        [np.where(all_nodes == node)[0][0] for node in list(df_high.index)]
    ]
    return nodes

def is_node_selection_by_node_info(selection_method):
    """Returns whether the selected method required node_info

    Arguments:
        selection_method {str} -- Node selection method chosen by user (args.selection_method)

    Returns:
        bool -- Boolean stating if the selected method requires the node_info
    """
    return (selection_method == SELECTION_METHOD_CORENESS or
            selection_method == SELECTION_METHOD_IN_DEG or
            selection_method == SELECTION_METHOD_OUT_DEG or
            selection_method == SELECTION_METHOD_PAGE_RANK)

def main():
    """Main execution steps in case of direct script execution

    This will create a grouped boxplot based on the similarity comparisons for a given dataset.
    """
    args = parse_args()

    all_nodes = None
    comparison_data = {}
    node_info = None

    # First check where to obtain the comparison data from
    if args.comparison_mode == COMPARISON_MODE_MANUAL:
        # For manual mode, the result files should be given directly
        print((f"\nUsing {COMPARISON_MODE_MANUAL} method to load comparisons\n"
               "First input file is considered to hold the nodes used for comparison\n"
               "The other ones should hold the comparison results for these nodes!"))
        all_nodes = np.load(args.inputs[0])
        assert len(all_nodes.shape) == 1, ("Shape of loaded nodes is not one-dimensional ",
                                           f"but {all_nodes.shape}. Maybe you forgot to ",
                                           "specify the node file first and instead put "
                                           "the first comparison file there?")
        print(f"\nLoaded {len(all_nodes)} nodes from {args.inputs[0]}")

        print(f"\nGathering {len(args.inputs[1:])} comparison files")
        for file_path in args.inputs[1:]:
            algorithm, data = read_comparison_data_from_file(file_path)
            comparison_data[algorithm] = data
        print(f"Found comparisons for algorithms: {', '.join(comparison_data.keys())}")
    else:
        # Otherwise calculate the comparison on-the-fly
        # If nodes should be selected by node-info, we first have to obtain them
        # in order to fix them as the nodes used for comparison calculations
        print(f"\nUsing {args.comparison_mode} method to create comparisons")
        if is_node_selection_by_node_info(args.selection_method):
            print((f"Since node selection method is set to {args.selection_method}, "
                   "the node sample for comparisons will include the nodes with highest/"
                   f"centered/lowest {args.selection_method}"))
            assert args.node_info is not None, ("Path to node_info file missing "
                                                f"for selection method {args.selection_method}. "
                                                "Please set it via --node-info")
            if args.sample_size < 3:
                print(("! Setting sampling size to 3 to fill all three groups with "
                       "at least one entry"))
                args.sample_size = 3
            if args.aggregation_size*3 < args.sample_size:
                print(("! Calculating the comparison for more nodes than those which will "
                       "be used in aggregation doesn't make much sense. Maybe you forgot to "
                       f"set the --aggregation-size. {args.sample_size} will be sampled "
                       f"but only --aggregation-size*3 = {args.aggregation_size*3} of them used."))
            if args.sample_size % 3 != 0:
                print(("! As there will be 3 extrema groups built, the sample size should "
                       f"be dividable by 3. Thus, only {args.sample_size - (args.sample_size%3)} "
                       "nodes will be sampled instead!"))
                args.sample_size = args.sample_size - (args.sample_size%3)
            node_info = parse_nodeinfo(args.node_info)
            node_info_sorted = node_info.sort_values(by=args.selection_method).index

            mid = int(len(node_info)/2)
            all_nodes = np.hstack((node_info_sorted[:int(args.sample_size/3)],
                                   node_info_sorted[mid-int(args.sample_size/6):
                                                    mid+1+int(args.sample_size/6)],
                                   node_info_sorted[-int(args.sample_size/3):]))
        # Continue with comparison execution!
        for folder in args.inputs:
            files = [file for file in listdir(folder) if isfile(join(folder, file))]

            print(f"\nInitializing comparison with {len(files)} files within {folder}")
            cmp = Comparison(folder, files)
            print(f"Executing {args.comparison_mode} comparison with:")
            cmp_result = None
            save = args.comparison_savepath is not None
            save_path = args.comparison_savepath
            if args.comparison_mode == COMPARISON_MODE_KNN:
                print(f"\tnodes: {all_nodes}\n\tsamples: {args.sample_size}\n\tk: {args.k}")
                cmp_result = cmp.k_nearest_neighbors(nodes=all_nodes,
                                                     samples=args.sample_size,
                                                     k=args.k,
                                                     save=save,
                                                     save_path=save_path)
                all_nodes = cmp_result["nodes"]
                comparison_data[folder] = cmp_result["overlaps"]
            elif args.comparison_mode == COMPARISON_MODE_JACCARD:
                print(f"\tnodes: {all_nodes}\n\tsamples: {args.sample_size}\n\tk: {args.k}")
                cmp_result = cmp.jaccard_score(nodes=all_nodes,
                                                     samples=args.sample_size,
                                                     k=args.k,
                                                     save=save,
                                                     save_path=save_path)
                all_nodes = cmp_result["nodes"]
                comparison_data[folder] = cmp_result["overlaps"]
            elif args.comparison_mode == COMPARISON_MODE_SO_CS:
                print(f"\tnodes: {all_nodes}\n\tsamples: {args.sample_size}\n\tk: {args.k}")
                cmp_result = cmp.second_order_cosine_similarity(nodes=all_nodes,
                                                                samples=args.sample_size,
                                                                k=args.k,
                                                                save=save,
                                                                save_path=save_path)
                all_nodes = cmp_result[0]
                comparison_data[folder] = cmp_result[1]
            elif args.comparison_mode == COMPARISON_MODE_PROC:
                assert save, (f"Comparison method {COMPARISON_MODE_PROC} requires "
                              "a file path given by --comparison-savepath. This "
                              "folder needs to contain a folder named procrustes_matrices")
                print(f"\tnodes: {all_nodes}\n\tk: {args.sample_size}")
                cmp.procrustes_transformation(save_path=save_path)
                cmp_result = cmp.procrustes_analysis(nodes=all_nodes,
                                                     k=args.sample_size,
                                                     save_path=save_path)
                all_nodes = cmp_result[0]
                comparison_data[folder] = cmp_result[1]
        print(f"Calculated comparisons for algorithms: {', '.join(comparison_data.keys())}")
        if save:
            print(f"Comparison results were written to {save_path}")

    # Next up: Choose a subset of nodes based on sampling, page_rank, ...
    print(f"\nSelect nodes via {args.selection_method}")
    node_ids = None
    key = "node" if args.aggregation_size <= 1 else "group"
    if args.selection_method == SELECTION_METHOD_MANUAL:
        # Use all available node ids of all_nodes
        print(f"Using all available nodes")
        node_ids = [range(len(all_nodes))]
    elif args.selection_method == SELECTION_METHOD_SAMPLING:
        print(f"Sampling {args.sample_size} nodes out of the available {len(all_nodes)} nodes")
        # Create random permutation of indices for nodes in all_nodes and pick the first ones
        node_ids = [np.random.permutation(np.arange(len(all_nodes)))[:args.sample_size]]
    elif is_node_selection_by_node_info(args.selection_method):
        print((f"Using {args.selection_method} to sample node extrema "
               f"groups of size {args.aggregation_size}"))
        if node_info is None:
            assert args.node_info is not None, ("Path to node_info file missing "
                                                f"for selection method {args.selection_method}. "
                                                "Please set it via --node-info")
            node_info = parse_nodeinfo(args.node_info)
        node_ids = choose_nodes_by_node_info(all_nodes, node_info, args.selection_method,
                                             args.aggregation_size)
    print(f"Final node ids choice: {[node for group in node_ids for node in group]} "
          f"organized in {len(node_ids)} group(s)")

    print("\nParsing comparisons to match required data format")
    parsed_data = parse_comparison_data(data=comparison_data,
                                        all_nodes=all_nodes,
                                        node_ids=node_ids)
    print(f"Resulting in {len(parsed_data)} total entries")

    print(f"\nPrinting plot to {args.output}")
    create_grouped_boxplots(parsed_data, args.output, key)
    print("Done!")

if __name__ == "__main__":
    main()
