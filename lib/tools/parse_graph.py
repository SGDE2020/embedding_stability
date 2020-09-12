"""Create networkx graphs from different file sources

Raises:
    ValueError: If provided path is neither a file, nor a folder
    ValueError: If the file format is unknown or cannot be parsed
"""
import csv
import json
import os

import networkx as nx
import scipy.sparse


def parse_graph(dataset, largest_cc=False):
    """
    Main entry point for all parser functions. In almost all cases, it should
    be sufficient to call this function with the location of the graph to be
    parsed (its directory or the precise filename, depending on the format).

    The nodes of the returned graphs are integers indexed from zero. If this is
    not the case in the source, the nodes will be renamed accordingly.

    If the dataset contains node labels, they will be stored as `class`
    attribute for each node.

    Parameters
    ----------
    dataset : str or pathlib.Path
        Directory of file from which the graph should be parsed. The function
        tries to guess the correct format of the dataset.
    largest_cc : bool
        Whether only the largest connected component (weakly connected
        component for directed graphs) or the whole graph should be returned.

    Returns
    -------
    nx.Graph
    """
    if os.path.isdir(dataset):
        return parse_graph_from_dir(dataset, largest_cc)
    elif os.path.isfile(dataset):
        if dataset.endswith(".csv"):
            with open(dataset, "rb") as dataset_file:
                return parse_graph_from_csv(dataset_file, largest_cc)
        else:
            return parse_graph_from_edgelist(dataset, largest_cc)
    else:
        raise ValueError("Invalid Dataset.")

def parse_graph_from_edgelist(path, largest_cc):
    graph = nx.read_edgelist(path, nodetype=int)

    graph_name = os.path.split(path)[-1].split(".")[0]
    return graph_name, graph

def parse_graph_from_csv(file, largest_cc=False, skip_first_line=True):
    """
    Parses a file from a SNAP Dataset and returns a networkx graph.

    Parameters
    ----------
    file : file
    largest_cc : bool
        Whether the complete graph in the dataset should be returned, or only
        its largest connected component.
    """
    graph_name = os.path.splitext(os.path.basename(file.name))[0]

    if skip_first_line:
        file.readline() # ignore first line `node_1,node_2`
    graph = nx.read_adjlist(file, delimiter=",", nodetype=int)

    # Change indices to start from zero if necessary
    if min(graph.nodes()) == 1:
        # print("Reducing indices by one to have them starting at zero.")
        graph = nx.relabel_nodes(graph, lambda x: x-1, copy=False)

    if largest_cc:
        original_size = len(graph)
        largest_component = max(nx.connected_components(graph), key=len)
        subgraph = nx.subgraph(graph, largest_component)
        graph = nx.Graph(subgraph)

        if len(graph) < original_size:
            print("Only considering largest connected component with %d nodes. Original graph had %d nodes." % (len(graph), original_size))
        else:
            print("Graph has only one connected component.")

    return graph_name, graph


def parse_graph_from_dir(path, largest_cc=False):
    """
    Parses a directory containing a graph from konect or the social computing
    repository and returns a networkx graph.

    Parameters
    ----------
    path : str
        Absolute path to the directory containing the graph.
    largest_cc : bool
        Whether the complete graph in the dataset should be returned, or only
        its largest connected component.
    """
    graph_name = os.path.dirname(path).split("/")[-1]
    #if "_" in os.path.dirname(path): # this was here for our first graphs, but really only leads to unnecessary long names
    #    graph_name += "_" + os.path.dirname(path).split("_")[-1]
    if os.path.isfile(path + "/out." + graph_name):
        graph = _parse_konect(path, graph_name)
        graph_name = graph_name.split("_")[-1]
    elif os.path.isfile(path + "/nodes.csv") and os.path.isfile(path + "groups.csv"):
        graph = _parse_social_computing_repository(path)
    elif os.path.isfile(path + "/" + graph_name + ".edgelist"):
        graph = _parse_edgelist(path, graph_name)
    elif os.path.isfile(path + "/" + graph_name + "-adjlist.txt"):
        graph = _parse_adjlist(path, graph_name)
    else:
        raise ValueError("Unknown graph format.")

    if largest_cc:
        original_size = len(graph)
        if graph.is_directed():
            largest_component = max(nx.weakly_connected_components(graph), key=len)
        else:
            largest_component = max(nx.connected_components(graph), key=len)

        subgraph = nx.subgraph(graph, largest_component)
        graph = nx.Graph(subgraph)
        if len(graph) < original_size:
            print("Only considering largest strongly connected component with %d nodes. Original graph had %d nodes." % (len(graph), original_size))
        else:
            print("Graph has only one connected component.")

    return graph_name, graph


def parse_densified_graph(file, largest_cc=False, skip_first_line=True):
    """
    Parses a file created by reconstruction (densified) as described in the LINE paper and returns a networkx graph.

    Parameters
    ----------
    file : file
    largest_cc : bool
        Whether the complete graph in the dataset should be returned, or only
        its largest connected component.
    """
    n = os.path.basename(file.name).split(".")
    graph_name = ".".join([n[0], n[-1]])

    graph = nx.read_weighted_edgelist(file, delimiter="\t", nodetype=int)

    # Change indices to start from zero if necessary
    if min(graph.nodes()) == 1:
        print("Reducing indices by one to have them starting at zero.")
        graph = nx.from_edgelist(edgelist=[(u-1, v-1) for (u, v) in graph.edges()],
                                 create_using=graph)

    if largest_cc:
        original_size = len(graph)
        largest_component = max(nx.connected_components(graph), key=len)
        subgraph = nx.subgraph(graph, largest_component)
        graph = nx.Graph(subgraph)

        if len(graph) < original_size:
            print("Only considering largest connected component with %d nodes. Original graph had %d nodes." % (len(graph), original_size))
        else:
            print("Graph has only one connected component.")

    return graph_name, graph

def prepend_N_E_to_edgelist(edgelist_path, new_edgelist_path):
    """Reads an entire edgelist and prepends number of nodes and edges as the first line

    Arguments:
        edgelist_path {str} -- Path to the edgelist
        new_edgelist_path {str} -- Path to where the new edgelist will be stored
    """
    N = 0
    E = 0
    lines = []
    with open(edgelist_path, "r") as file:
        for line in file:
            split = line.split(" ")
            N = max(N, int(split[0]), int(split[1]))
            lines.append(split[0] + " " + split[1] + "\n")
            E = E + 1
        N = N + 1

    with open(new_edgelist_path, 'w') as file:
        file.write(f"{N} {E}\n")
        file.writelines(lines)

def _parse_adjlist(path, graph_name):
    graph = nx.Graph()
    name_to_id = dict()

    # The nodes are strings in the reddit dataset. These require significantly
    # more memory than integers, so we rename nodes to integers here.
    with open(path + "/" + graph_name + "-adjlist.txt", "r") as adjlist_file:
        for line in adjlist_file:
            if line.startswith("#"):
                # lines starting with '#' are meta information not corresponding to nodes in the graph
                continue
            nodes = map(lambda s: name_to_id.setdefault(s, len(name_to_id)), line.replace("\n", "").split(" "))
            u = next(nodes)
            for v in nodes:
                graph.add_edge(u, v)

    if os.path.isfile(path + "/" + graph_name + "-class_map.json"):
        print("Reading node classes...")
        with open(path + "/" + graph_name + "-class_map.json", "r") as label_file:
            labels = json.load(label_file)
            for node, label in labels.items():
                if node in name_to_id: # ignore classes for unknown nodes
                    graph.nodes[name_to_id[node]]["class"] = label

    return graph

def _parse_edgelist(path, graph_name, data=False):
    graph = nx.read_edgelist(path + "/" + graph_name + ".edgelist", nodetype=int, data=data)

    if os.path.isfile(path + "/" + graph_name + ".labels.npz"):
        label_matrix = scipy.sparse.load_npz(path + "/" + graph_name + ".labels.npz").todense()
        for n in graph:
            graph.nodes[n]["class"] = label_matrix[n]

    return graph


def _parse_konect(path, graph_name):
    with open(path + "/out." + graph_name, "rb") as edge_file:
        # first line contains if graph is connected or not
        if "asym" in edge_file.readline().decode("utf-8"):
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()

        edge_file.readline() # ignore second line with graph sizes

        graph = nx.read_edgelist(edge_file, nodetype=int, create_using=graph)

        # Indices should start at 0
        if min(graph.nodes()) == 1:
            print("Reducing indices by one to have them starting at zero.")
            graph = nx.relabel_nodes(graph, lambda x: x-1, copy=False)

    with open(path + "/ent." + graph_name + ".class.name") as class_file:
        for i, class_name in enumerate(class_file):
            # We only want to consider the top level category, because
            # otherwise the number of categories explodes. This is only
            # relevant for Cora and has no effect for all other datasets we
            # consider.
            class_name = class_name.strip("/").split("/")[0]

            graph.nodes[i]["class"] = class_name

    return graph

def _parse_social_computing_repository(path):
    graph = nx.Graph()

    with open(path + "/nodes.csv") as nodes_file:
        graph.add_nodes_from(map(int, nodes_file))

    with open(path + "/edges.csv", newline="") as edges_file:
        edges_reader = csv.reader(edges_file, delimiter=",")
        for edge in edges_reader:
            graph.add_edge(int(edge[0]), int(edge[1]))

    with open(path + "/group-edges.csv", newline="") as groups_file:
        groups_reader = csv.reader(groups_file, delimiter=",")
        for group_label in groups_reader:
            graph.nodes[int(group_label[0])]["class"] = int(group_label[1])

    # Indices should start at 0
    if min(graph.nodes()) == 1:
        print("Reducing indices by one to have them starting at zero.")
        graph = nx.relabel_nodes(graph, lambda x: x-1, copy=False)

    return graph
