import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from lib.tools.embedding import read_embedding

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("embedding_file", type=argparse.FileType("r"))
arg_parser.add_argument("-transform", type=str.upper, choices=["NONE", "PCA", "TSNE"], default="NONE")
arg_parser.add_argument('-save', type=str)
args = arg_parser.parse_args()

node_embeddings = read_embedding(args.embedding_file)

print("Transforming embedding vectors into 2D.")
if args.transform == "PCA":
    transform = PCA(n_components=2)
    node_embeddings_2d = transform.fit_transform(node_embeddings)
elif args.transform == "TSNE":
    transform = TSNE(n_components=2)
    node_embeddings_2d = transform.fit_transform(node_embeddings)
elif args.transform == "NONE":
    if node_embeddings.shape[1] != 2:
        exit("Embedding is " + str(node_embeddings.shape[1]) + "-dimensional. Transformation must be used for higher-dimensional embeddings.")

    node_embeddings_2d = node_embeddings

plt.figure()
plt.scatter(node_embeddings_2d[:,0], node_embeddings_2d[:,1])
plt.title("{0} visualization of {1}".format(args.transform, args.embedding_file.name))

if args.save:
    plt.savefig(args.save, bbox_inches="tight")

plt.show()
