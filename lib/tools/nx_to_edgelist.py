import os
import networkx as nx

def nx_to_weighted_edgelist(graph: nx.Graph, edge_list_file: str):
        """ 
        Adds a third column to the edge list for weights with value 1 if not provided. 
        nx.write_weighted_edgelist will not write a weights column if weights are not specified in the original graph.
        """
        edges = list(graph.edges(data=True))
        edges_str = None
        if bool(edges[0][2]) is True:  # check whether dict is not empty
                edges_str = "\n".join(list(map(lambda edge: str(edge[0]) + " " + 
                                str(edge[1]) + " " + str(edge[2]["weight"]), edges)))
        else:
                edges_str = "\n".join(list(map(lambda edge: str(edge[0]) + " " + str(edge[1]) + " 1", edges)))

        with open(edge_list_file, "w+") as file:
                file.write(edges_str)
