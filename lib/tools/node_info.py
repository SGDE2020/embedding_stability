"""DEPRECATED DUE TO GRAPH_TOOLS-BASED get_node_info.py SCRIPT"""

from lib.tools.parse_graph import parse_graph_from_csv, parse_graph_from_dir, parse_densified_graph, parse_graph
import networkx as nx
import numpy as np
import pandas as pd
import argparse

SOURCE_SNAP = "snap"
SOURCE_KONECT = "konect"
SOURCE_LINE = "line"
SOURCE_ASU = "asu"

def create_nodeinfo(source:str, graph_path:str, save_path:str, file_name:str):
    """
    Creates a file that contains a space seperated table, in which additional information to specific nodes is saved.
    The format is node | in_degree | out_degree | page_rank | coreness

    source: str that specifies if graph data is "konect" or "snap"
    """
    if source in ["snap", "blog"]:
        with open(graph_path, "rb") as f:
            graph_name, G = parse_graph_from_csv(f)
    elif source in ["konect", "wiki", "protein", "synth"] :
        graph_name, G = parse_graph(graph_path)
    elif source == "line":
        with open(graph_path, "rb") as f:
            graph_name, G = parse_densified_graph(f)
    else:
        print("Unknown source of graph.")
        return

    G = G.to_directed()
    with open(save_path + file_name, "w") as f:
        nodes = [str(n) for n in G]
        in_deg = [str(deg) for _, deg in G.in_degree()]
        out_deg = [str(deg) for _, deg in G.out_degree()]
        page_rank = [str(rank) for _, rank in nx.pagerank(G).items()]
        # Keep coreness at the end because the graph has to be modified
        G.remove_edges_from(nx.selfloop_edges(G))
        coreness = [str(cn) for _, cn in nx.core_number(G).items()]
        lines = [" ".join(data) for data in sorted(zip(nodes, in_deg, out_deg, page_rank, coreness), key=lambda x: int(x[0]))]
        lines.insert(0, " ".join(["node", "in_deg", "out_deg", "page_rank", "coreness"]))
        f.write("\n".join(lines))
        print(save_path + file_name, "was created.")

def parse_nodeinfo(path):
    """
    Parses a info file created by create_nodeinfo and returns a pandas dataframe where colums correspond to the information specified in create_nodeinfo.
    The dataframe is indexed by node.
    """
    df = pd.read_csv(path, sep=" ", header=0, index_col=0)
    if len(df.columns) == 5:  # CHANGE THIS WHEN NEW FEATURES ARE ADDED
        return df
    else:
        print("Unexpected number of colums read. Is the node info file up to date?")
        return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates a node_info file for a given graph.")

    parser.add_argument("source", type=str,
                        choices=[
                            SOURCE_SNAP,
                            SOURCE_ASU,
                            SOURCE_KONECT,
                            SOURCE_LINE
                        ],
                        help="Sets the source type of the graph")
    parser.add_argument("graph_path", type=str,
                        help="Specifies the path to the graph dataset")
    parser.add_argument("output_path", type=str,
                        help="The file the results should be written to")
    args = parser.parse_args()
    create_nodeinfo(args.source, args.graph_path, args.output_path, "")