"""
Script used to create exploratory boxplots.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse as ap
import os
import numpy as np

SIMILARITY_MEASURE_COSSIM = "cossim"
SIMILARITY_MEASURE_LINPROC = "linproc"
SIMILARITY_MEASURE_KNN = "knn"
SIMILARITY_MEASURE_JACCARD = "jaccard"
SIMILARITY_MEASURE_RANKS = "ranks"
SIMILARITY_MEASURE_RNS = "rns"
SIMILARITY_MEASURE_ANGDIV = "angdiv"
SIMILARITY_MEASURE_2NDCOS = "2ndcos"

SIMILARITY_MEASURE_FILE_ENDINGS = {
    SIMILARITY_MEASURE_COSSIM: "_aligned_cossim.npy",
    SIMILARITY_MEASURE_LINPROC: "_linproc_cossim.npy",
    SIMILARITY_MEASURE_KNN: "_20nn_overlap.npy",
    SIMILARITY_MEASURE_JACCARD: "_20nn_jaccard.npy",
    SIMILARITY_MEASURE_RANKS: "_20nn_ranks.npy",
    SIMILARITY_MEASURE_RNS: "_rns.npy",
    SIMILARITY_MEASURE_ANGDIV: "20nn_angdiv.npy",
    SIMILARITY_MEASURE_2NDCOS: "_20nn_2nd_order_cossim.npy",
}


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-i", "--input-folder", type=str)
    # parser.add_argument("-d", "--datasets", type=str)
    # parser.add_argument("-a", "--algorithms", type=str)
    parser.add_argument("-s", "--similarity-measure", type=str,
                        choices=SIMILARITY_MEASURE_FILE_ENDINGS.keys())
    return parser.parse_args()


def run(similarity_measure, input_folder, output=None, do_plot=False):
    """
    Create boxplots of similarity measures for different graphs and combine them in one plot.

    Args:
        similarity_measure: str, one out of four different similarity measures
        input_folder: str, path to a folder with saved experiment results
        output: str, path in which the image will be saved

    Returns:
        data_frame: pandas dataFrame that holds all experiment values
    """
    file_ending = SIMILARITY_MEASURE_FILE_ENDINGS[similarity_measure]

    # Dict to later create a pandas.DataFrame from
    data = {"similarity": [],
            "algorithm": [],
            "graph": [],
            "experiment": []}

    for file_name in os.listdir(input_folder):
        # Filter out synthetic graph results
        if file_name.endswith(file_ending) and "_ws_" not in file_name and "_ba_" not in file_name:
            file_name_short = file_name[:-len(file_ending)]
            file_name_split = file_name_short.split("_")
            print(f"Including {file_name}")
            similarity_input = np.load(input_folder + '/' + file_name)

            # Get algorithm and dataset from filename:
            # <algorithm>_<graph>_<file_ending>
            algorithm = file_name_split[0]
            graph = file_name_short[len(algorithm) + 1:]

            # Fill dict for DataFrame
            for similarity_value in similarity_input.flatten():
                data["algorithm"].append(algorithm)
                data["graph"].append(graph)
                if similarity_measure == SIMILARITY_MEASURE_COSSIM or similarity_measure == SIMILARITY_MEASURE_LINPROC:
                    # Convert the measure to cosine similarity from cosine distance
                    similarity_value = 1 - similarity_value
                data["similarity"].append(similarity_value)
                data["experiment"].append(similarity_measure)

            print(f"\talgorithm: {algorithm}")
            print(f"\tgraph: {graph}")
            print(f"\t{similarity_input.shape[0] * similarity_input.shape[1]} similarity values")

    print(f"Writing {len(data['similarity'])} total similarity values to DataFrame...")
    data_frame = pd.DataFrame(data=data)
    if do_plot:
        print("Plotting...")
        figure = plt.figure()
        axis = sns.boxplot(x="graph", y="similarity", hue="algorithm", data=data_frame, fliersize=0.1)

        figure.add_axes(axis)
        plt.xticks(rotation=315)
        if output is not None:
            print(f"Saving to {output}")
            figure.savefig(output)
    print("Done!")
    return data_frame


if __name__ == "__main__":
    args = parse_args()
    run(args.similarity_measure, args.input_folder, args.output)
