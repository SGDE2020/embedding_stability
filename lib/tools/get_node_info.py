"""
This script is used to compute node properties, e.g. in-degree, coreness etc.
The result are saved as .txt file.
Example usage (requiring the graph saved as .graphml in the folder specified by -p):
    python3 get_node_info.py -g protein -p ../train/data/protein/ -s no -o ../similarity_tests/node_info/
"""

import numpy as np
import os
from graph_tool.all import *
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser()

# Parse algorithms
parser.add_argument("-g", "--gname", default ="")
parser.add_argument("-p", "--path")
parser.add_argument("-s", "--synth", default = "no")
parser.add_argument("-o", "--outpath", default = "nodeinfo/")

args = parser.parse_args()

# node info header
h = "node in_deg out_deg page_rank coreness betweenness closeness"

def get_nodeinfo_graphml(path, gname, outpath):
    """
    Loads a graph in .graphml format and computes degrees, pagerank, coreness and betweenness centrality
    for all nodes. The result is saved as a space separated table in .txt format.

    Args:
        path: str, path to directory where graph is saved
        gname: str, graph file name
        outpath: str, path to directory where result will be saved
    """

    G = graph_tool.load_graph(path + gname + ".graphml")
    N = G.num_vertices()
    idg = G.get_in_degrees(range(N))
    odg = G.get_out_degrees(range(N))
    pr = graph_tool.centrality.pagerank(G).get_array()
    kcs = graph_tool.topology.kcore_decomposition(G).get_array()
    bc,_ = graph_tool.centrality.betweenness(G)
    cc = graph_tool.centrality.closeness(G).get_array()

    fname = outpath + gname + ".node_info"
    stats = np.vstack([np.array(range(N)),idg,odg,pr,kcs,bc.get_array(),cc]).T

    print(f"Saving node-info to {fname}")
    np.savetxt(fname, stats, header=h, fmt=["%1i","%1i","%1i","%1e","%1i","%1e","%1e"], comments='')


def get_nodeinfo_edgelist(path, fname, outpath):
    """
    Loads a graph in .edgelist format and computes degrees, pagerank, coreness and betweenness centrality
    for all nodes. The result is saved as a space separated table in .txt format.
    """

    N = int(fname.split('_')[1][1:])

    G = Graph(directed=False)
    G.add_vertex(n=N)

    print(fname + " opened")
    edge_list = []

    with open(path+fname) as f:
        for line in f:
            s = line.split(' ')
            edge_list.append((int(s[0]), int(s[1])))
    f.close()
    print("read-in completed")

    G.add_edge_list(edge_list)
    idg = G.get_in_degrees(range(N))
    odg = G.get_out_degrees(range(N))
    pr = graph_tool.centrality.pagerank(G).get_array()
    kcs = graph_tool.topology.kcore_decomposition(G).get_array()
    bc,_ = graph_tool.centrality.betweenness(G)
    cc = graph_tool.centrality.closeness(G).get_array()

    stats = np.vstack([np.array(range(N)),idg,odg,pr,kcs,bc.get_array(),cc]).T
    fname = outpath + fname.split('.')[0] + ".node_info"

    print(f"Saving node-info to {fname}")
    np.savetxt(fname, stats, header=h, fmt=["%1i","%1i","%1i","%1e","%1i","%1e","%1e"])

    print("stats saved")


def run(path, gname, synth, outpath):

    if synth == "yes":
        for fname in os.listdir(path):
            get_nodeinfo_edgelist(path,fname, outpath)
    else:
        get_nodeinfo_graphml(path,gname, outpath)

if __name__ == "__main__":
    run(args.path, args.gname, args.synth, args.outpath)
