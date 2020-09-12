""" Create tables of aggregated similarity measures.
See parse_args() to see how to use or use the 'tables' IPython notebook. """

import argparse

import numpy as np
import pandas as pd

from lib.tools.comparison import Comparison

COMPARISON_MODE_MEAN_OF_MEANS = "mom"
COMPARISON_MODE_STD_OF_MEANS = "som"
COMPARISON_MODE_MEAN_OF_STDS = "mos"
COMPARISON_MODE_KNN = "knn"
COMPARISON_MODE_SO_CS = "20nn_2nd_order_cossim"
COMPARISON_MODE_PROC = "procrustes"
COMPARISON_MODE_LINPROC = "linproc"

def parse_args():
    arg_parser = argparse.ArgumentParser(
        description=("Creates a table for each given comparison method "
                     "comparing the results for pairs of embedding "
                     "algorithms and datasets")
    )

    arg_parser.add_argument("-o", "--output", type=str, help="Specifies output destination")
    arg_parser.add_argument("comparison_mode", type=str,
                            choices=[
                                COMPARISON_MODE_MEAN_OF_MEANS,
                                COMPARISON_MODE_STD_OF_MEANS,
                                COMPARISON_MODE_MEAN_OF_STDS,
                                COMPARISON_MODE_SO_CS,
                                COMPARISON_MODE_PROC,
                                COMPARISON_MODE_LINPROC],
                            help=(""))
    arg_parser.add_argument("--algorithms", "-a", type=str, nargs="+", help="")
    arg_parser.add_argument("--datasets", "-d", type=str, nargs="+", help="")
    arg_parser.add_argument("inputs", type=str, nargs="+", help=(""))
    arg_parser.add_argument("--is-distance", action="store_true")
    return arg_parser.parse_args()

def run(algorithms, datasets, inputs, comparison_mode, is_distance, output):

    table_shape = (len(algorithms), len(datasets))
    results = np.empty(table_shape)

    inputs = np.reshape(inputs, table_shape)

    for i in range(len(algorithms)):
        for j in range(len(datasets)):
            comparison_result = None
            measures = np.load(inputs[i][j])
            print(f"Input: {inputs[i][j]}, Shape: {measures.shape}")
            if comparison_mode == COMPARISON_MODE_MEAN_OF_MEANS:
                comparison_result = np.mean([node.mean() for node in measures])
                if is_distance:
                    comparison_result = 1 - comparison_result
            elif comparison_mode == COMPARISON_MODE_STD_OF_MEANS:
                comparison_result = np.std([node.mean() for node in measures])
            elif comparison_mode == COMPARISON_MODE_MEAN_OF_STDS:
                comparison_result = np.mean([node.std() for node in measures])
            results[i][j] = comparison_result

    frame = pd.DataFrame(results, index=algorithms, columns=datasets)
    print(frame)
    index_dict = {"line":"LINE", "hope":"HOPE", "sdne":"SDNE", "graphsage":"GraphSAGE", "node2vec":"node2vec"}
    column_dict = {"cora":"Cora", "facebook":"Facebook", "blogcatalog": "BlogCatalog", "protein":"Protein",
                "wikipedia":"Wikipedia"}
    frame.rename(index=index_dict, inplace=True)
    frame.rename(columns=column_dict, inplace=True)
    frame.to_csv(output)

if __name__ == "__main__":
    args = parse_args()
    run(args.algorithms, args.datasets, args.inputs, args.comparison_mode, args.is_distance, args.output)
