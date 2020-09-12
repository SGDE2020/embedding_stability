"""Script to transform edgelists and gramph files to formats required by SDNE embedding algorithm
"""
import networkx as nx
import scipy.io as sio
import os

def translate_graphml(path, infile):
    """
    translates .graphml file into .mat file
    """

    G = nx.read_graphml(path + infile)
    outfile = infile.split('.')[0] + ".mat"
    A = nx.adjacency_matrix(G)
    sio.savemat(path+outfile, {"graph_sparse": A})


def translate_edgelist(path, infile):
    """Translates edgelist to correct format

    N and E must be given in the first row of the edgelist.
    New edgelist will be written to same folder with .txt ending.

    Arguments:
        path {str} -- Path to folder of given edgelist
        infile {str} -- Filename of edgelist
    """
    N = int(infile.split('_')[1][1:])

    e = 0
    with open(path+infile) as f:
        for line in f: e += 1
    f.close()

    outfile = infile.split('.')[0] + ".txt"

    with open(path+outfile, "w") as outf:
        outf.write(str(N) + " " + str(e) + "\n")
        with open(path+infile) as inf:
            for line in inf:
                s = line.split(' ')
                outf.write(s[0] + " " + s[1] + "\n")
    inf.close()
    outf.close()
    print(outfile + " written")


def run(path, gname, synth):
    """Translates given edgelist or graphml file to correct format

    Arguments:
        path {str} -- Path to folder of existing file
        gname {str} -- Graph name, required for graphml input format (synth != "yes")
        synth {str} -- "yes" if synthetical graph (all edgelists in path will be transformed)
    """
    if synth == "yes":
        for infile in os.listdir(path):
            translate_edgelist(path,infile)
    else:
        translate_graphml(path,gname)
