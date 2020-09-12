"""
Module for creating line plots of stability measures.
Example usage: python3 plots.py -d results/line_cora_overlap.npy [more files correspong to other algorithms but the same graph]
                        -i node_info/cora.node_info
                        -w 20 20 30 9
                        -s mylineplots
"""
import argparse
from bisect import insort, bisect_left
from collections import deque
from itertools import islice, combinations
import os
import pickle
from time import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

def plot_distribution(info_dir="/node_info", save_dir="/plots/"):
    for f in os.listdir(info_dir):
        if f.endswith(".node_info") is False:
            continue
        print(f)
        info = parse_nodeinfo(info_dir + f)

        fig, axes = plt.subplots(2, 2, figsize=(7, 7))
        #plt.tight_layout()
        axes[0, 0].title.set_text("in degree distribution")
        axes[0, 1].title.set_text("out degree distribution")
        sns.distplot(info.iloc[:, 1], kde=False, ax=axes[0, 0])
        sns.distplot(info.iloc[:, 2], kde=False, ax=axes[0, 1])
        sns.boxplot(info.iloc[:, 1], ax=axes[1, 0])
        sns.boxplot(info.iloc[:, 2], ax=axes[1, 1])

        fig.savefig(save_dir + f + "_degdistr.jpg")

def plot_scatterlike(nodes, y,  info, descr="", err=None, file_name=None, save_path=None):
    """Scatterplot (with errorbars if err!=None) of experiment results"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes[0, 0].title.set_text(f"{descr} over in degree")
    axes[0, 1].title.set_text(f"{descr} over out degree")
    axes[1, 0].title.set_text(f"{descr} over in+out degree")
    axes[1, 1].title.set_text(f"{descr} over page rank")
    axes[2, 0].title.set_text(f"{descr} over coreness")

    df = info
    axes[0, 0].errorbar(df.loc[nodes, "in_deg"], y, err, fmt="o")
    axes[0, 1].errorbar(df.loc[nodes, "out_deg"], y, err, fmt="o")
    axes[1, 0].errorbar(df.loc[nodes, "in_deg"] + df.loc[nodes, "out_deg"], y, err, fmt="o")
    axes[1, 1].set_xscale("log")
    axes[1, 1].errorbar(df.loc[nodes, "page_rank"], y, err, fmt="o")
    axes[2, 0].errorbar(df.loc[nodes, "coreness"], y, err, fmt="o")

    if save_path is None:
        save_path = os.getcwd()
    if file_name is None:
        file_name = str(time())
    fig.savefig(file_name + ".jpg")

def parse_args():
    arg_parser = argparse.ArgumentParser(description="Lineplots of similarity test results")
    arg_parser.add_argument("-w", "--window-sizes", type=int, nargs="+", default=30,
                            help="Window size(s) for moving average/median filter")
    arg_parser.add_argument("-f", "--filter-type", type=str, choices=["mean", "median", "none"], default="median",
                            help="What kind of filter to use")
    arg_parser.add_argument("-d", "--data", type=str, nargs="+", help="file(s) with results of a stability measure/nearest neighbors")
    arg_parser.add_argument("-i", "--info-file", type=str, help=".node_info file for the graph")
    arg_parser.add_argument("-s", "--save-path", type=str, default="./", help="Where to save the plots")
    arg_parser.add_argument("-y", "--ylabel", type=str, help="Label of y axis")
    arg_parser.add_argument("-x", "--xscale", type=str, default="log", help="Scaling of x axis")
    arg_parser.add_argument("-t", "--title", type=str, default="", help="Title of plot")
    arg_parser.add_argument("-k", type=int, default=20, help="How much of a raw query matrix to use")
    arg_parser.add_argument("-m", "--mode", type=str, choices=["knn", "2ndcos"], help="Which metric is plotted. Important if a raw query matrix is used")
    arg_parser.add_argument("-e", "--emb-dir", type=str, help="Directory where embeddings are saved")
    arg_parser.add_argument("--ylims", type=float, nargs="+", default=[0, 1])
    return arg_parser.parse_args()

def moving_average(seq, window_size):
    cumsum_vec = np.cumsum(np.insert(seq, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

def moving_median(seq, window_size):
    """Contributed by Peter Otten"""
    seq = iter(seq)
    d = deque()
    s = []
    result = []
    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        if len(d) % 2 == 0:
            result.append(0.5 * (s[len(d) // 2] + s[len(d) // 2 - 1]))
        else:
            result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    return result

def build_dataframe(experiment_data, info_frame, experiment_name, xscale="log", median=True):
    """Create a DataFrame out of a numpy array and a nodeinfo DataFrame
    Params:
        experiment_data: dict of np.ndarray, test results keyed by algorithm
        info_frame: pd.DataFrame, contains node property values (e.g. PageRank)
        experiment_name: str, identifier for kind of experiment
        xscale: str, if "log", scale node property values logarithmically
        median: bool, whether the median of mean similarity or mean of mean similarity is taken
                     as similarity value for a node over all embeddings
    Returns:
        pd.DataFrame, one row for every node for every algorithm for every statistic/property
    """

    di = {"algorithm": [], "statistic": [], "x": [], "y":[]}

    if median is True:
        agg_func = np.median
    else:
        agg_func = np.mean
    if experiment_name == "procrustes_cossim"  or experiment_name == "linproc_cossim":
        for algo in experiment_data.keys():
            for column in info_frame.columns:
                print(algo, column)
                if column not in ["page_rank", "coreness", "closeness"]:
                    continue
                for val in np.sort(info_frame[column].unique()):
                    nodes = info_frame[info_frame[column] == val].index
                    di["algorithm"].append(algo)
                    di["statistic"].append(column)
                    di["x"].append(val)
                    cossims = 1 - experiment_data[algo][:, nodes].mean(axis=0)
                    di["y"].append(agg_func(cossims))
    else:
        for algo in experiment_data.keys():
            for column in info_frame.columns:
                # Only plot PageRank and coreness
                if column not in ["page_rank", "coreness", "closeness"]:
                    continue
                print(algo, column)
                for val in np.sort(info_frame[column].unique()):
                    nodes = info_frame[info_frame[column] == val].index
                    di["algorithm"].append(algo)
                    di["statistic"].append(column)
                    di["x"].append(val)
                    di["y"].append(agg_func(experiment_data[algo][:, nodes].mean(axis=0)))
                    #di["y"].append(agg_func(experiment_data[algo][:threshold, nodes].mean(axis=0)))
    pf = pd.DataFrame(di)
    return pf

def _calc_filter(filter_type, data_series, window_size):
    if filter_type == "median":
        return moving_median(data_series.values, window_size)
    elif filter_type == "mean":
        return data_series.rolling(window_size).mean().values
        #return moving_average(data, window_size).tolist()
    raise ValueError("Unexpected filter type:", filter_type)

def calc_filter(filter_type, data_frame, window_size):
    """Calculate filter values for columns of a DataFrame"""
    filter_values = {}
    for algo in data_frame.algorithm.unique():
        filter_values[algo] = {}
        algo_frame = data_frame[data_frame.algorithm == algo]
        for statistic in data_frame.statistic.unique():
            filter_values[algo][statistic] = _calc_filter(filter_type,
                                                          algo_frame.loc[algo_frame.statistic == statistic, "y"],
                                                          window_size[statistic])
    return filter_values

def set_ylimits(axes, limits):
    for ax in axes:
        if len(limits) == 2:
            ax.set_ylim(bottom=limits[0], top=limits[1])
        else:
            ax.set_ylim(bottom=limits[0])

def plot_line(args_data, args_mode, args_info_file, args_xscale, args_window_sizes, args_k, args_emb_dir,
            args_filter_type, args_ylabel, args_ylims, args_save_path, agg_median=False):
    """
    Creates line plots of similarity measures over node properties.
    The lines can be smoothed with a moving median or moving average filter. Window size can be specified seperately for every node property.

    Args:
        args_data: list, list of paths of experiment results
        args_mode: str, whether "knn" or "2ndcos" is plotted
        args_info_file: str, path to info file of graph
        args_xscale: str, matplotlib argument for x-axis scaling, e.g. 'log'
        args_window_sizes: list, ints ordered by corresponding node property
        args_k: int, not used
        args_emb_dir: str, not used
        args_filter_type: str, "mean", "median" or "none", the type of filter for the plots
        args_ylabel: str, ylabel of plot
        args_ylims: list of floats, y-axis limits, if length is 1, only the lower limit is set
        args_save_path: str, path where to save the plot
        agg_median: bool, whether on the per node level median or mean will be used to aggregate values of different embeddings

    Returns:
        pandas.DataFrame that is used for plotting

    """

    t1 = time()
    mf_frame = None
    # Build a dict that holds the experiment results index by algorithm
    data = {}
    experiment_name = None
    for d in args_data:
        if "procrustes_cossim" in d:  # we need this to correctly compute the similarity from the distance matrix
            experiment_name = "procrustes_cossim"
        if d.endswith(".npy"):
            data[os.path.split(d)[-1].split("_")[0]] = np.load(d)
        else:
            raise ValueError("Data provided in args_data must be .npy files")

    # Load graph statistics
    info_frame = pd.read_csv(args_info_file, sep=" ", header=0, index_col=0)

    # Build data frame used for plotting
    pf = build_dataframe(data, info_frame, experiment_name, xscale=args_xscale, median=agg_median)

    # Plot seaborn.FacetGrid
    # Check whether the correct number of windows are specified (one for each statistic)

    if len(args_window_sizes) != len(pf.statistic.unique()):
        print(len(args_window_sizes), pf.statistic.unique())
        args_window_sizes = [args_window_sizes[0] for i in range(len(pf.statistic.unique()))]
        print("\nWarning: Unexpected number of window sizes. First element will be used.\n")
    window_sizes = dict(zip(pf.statistic.unique(), args_window_sizes))

    # Compute the filter values if needed
    filter_values = {}
    if args_filter_type != "none":
            filter_values = calc_filter(args_filter_type, pf, window_sizes)

    # If only a single algorithm is specified, create a detailed plot
    if len(args_data) == 1:
        grid = sns.FacetGrid(pf, col="statistic", hue="statistic",
                             col_wrap=4, height=4, sharex=False, palette="Pastel2")  # Change color for statistic here
        grid.set_ylabels(args_ylabel)
        grid.map(plt.plot, "x", "y")
        set_ylimits(grid.axes, args_ylims)
        if bool(filter_values):
            axes = grid.axes
            cmap = mpl.cm.get_cmap("Dark2")  # Change color for moving average/median here
            for i, statistic in enumerate(pf.statistic.unique()):
                axes[i].plot(pf.loc[pf.statistic == statistic, "x"].values,
                             list(filter_values.values())[0][statistic], color=cmap.colors[i])
                axes[i].set_xscale(args_xscale)
                for tick in axes[i].get_xticklabels():
                    tick.set_rotation(-45)

    # Multiple files are specified. Hence, multiple algorithms will be plotted. Only show the filter values
    else:
        if args_filter_type == "none":
            grid = sns.FacetGrid(pf, col="statistic", hue="algorithm", col_wrap=2,
                            height=4, sharex=False, palette="Pastel2")
            grid.set_ylabels(args_ylabel)
            grid.map(plt.plot, "x", "y")
        else:
            # Only plot the moving filter value. We need a dataframe with the filter values instead of raw data
            mf_frame = pd.DataFrame(columns=pf.columns)
            for algo, algo_values in filter_values.items():
                algo_frame = pf[pf.algorithm == algo].copy()
                for statistic in pf.statistic.unique():
                    algo_frame.loc[algo_frame.statistic == statistic, "y"] = algo_values[statistic]
                mf_frame = mf_frame.append(algo_frame)

            # Create the plot
            grid = sns.FacetGrid(mf_frame, col="statistic", hue="algorithm", height=4, col_wrap=2, sharex=False, palette="Dark2")
            grid.map(plt.plot, "x", "y")
            axes = grid.axes
            for i, _ in enumerate(axes.flat):
                axes[i].set_xscale(args_xscale)
                for tick in axes[i].get_xticklabels():
                    tick.set_rotation(-45)
        set_ylimits(grid.axes, args_ylims)
        grid.add_legend()
        grid.set_ylabels(args_ylabel)

    grid.savefig(args_save_path)
    print(f"Plotting took {time() - t1} seconds.")
    if mf_frame is not None:
        return mf_frame
    else:
        return pf

if __name__ == "__main__":
    args = parse_args()
    plot_line(args.data, args.mode, args.info_file, args.xscale, args.window_sizes, args.k, args.emb_dir, args.filter_type, args.ylabel, args.ylims, args.save_path)
