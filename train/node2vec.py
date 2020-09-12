import argparse
import os

import networkx as nx
import numpy as np
import random
from gensim.models import Word2Vec

from lib.node2vec_embedding.randomwalk import Graph
from lib.tools.embedding import save_embedding
from lib.tools import parse_graph


def generate(node2vec_graph, dimensions=128, walk_length=80, walk_number=10, p=1.0, q=1.0, seed=None):
    if seed is not None:
        workers = 1
    else:
        seed = 1
        workers = 8

    if isinstance(node2vec_graph, nx.Graph):
        print("Notice: When calling node2vec multiple times on the same graph,"
              "you should pass a node2vec graph instance. This allows to share"
              "preprocessed data between runs.")
        node2vec_graph = Graph(node2vec_graph, p, q, use_alias=False, seed=seed)

    walks = node2vec_graph.simulate_walks(walk_number, walk_length)
    model = Word2Vec(list(walks), size=dimensions, window=10, min_count=0, sg=1, workers=workers, iter=1, seed=seed)

    embedding = np.empty((len(node2vec_graph.G), dimensions))
    for j, node in enumerate(node2vec_graph.G):
        embedding[j] = model.wv[str(node)]

    return embedding


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Runs the node2vec representation learning algorithm on the given graph.")
    arg_parser.add_argument("dataset", nargs="+", type=str, help="Path to the dataset(s) for which embeddings should be generated.")
    arg_parser.add_argument("-largest-cc", action="store_true", help="States whether only the largest (weakly) connected component will be used, which might be useful when dealing with disconnected graphs.")
    arg_parser.add_argument("-num-embeddings", type=int, default=1, help="Number of embeddings that should be generated for each graph.")
    arg_parser.add_argument("-no-alias", action="store_true", help="Turn of alias preprocessing, e.g. when running out of memory otherwise.")
    arg_parser.add_argument("--seed", type=int, default=None, help="Sets random seed (useful for reproducibility). If this is set, the environmental variable PYTHONHASHSEED needs to be set as well in order to ensure a deterministic outcome!")
    arg_parser.add_argument("--output", type=str, help="Set absolut output filename. Only possible if generating only one embedding")

    # standard parameters from original node2vec implementation
    arg_parser.add_argument("-dimensions", type=int, default=128)
    arg_parser.add_argument("-walk-length", type=int, default=80)
    arg_parser.add_argument("-walk-number", type=int, default=10)
    arg_parser.add_argument("-p", type=float, default=1.0)
    arg_parser.add_argument("-q", type=float, default=1.0)

    args = arg_parser.parse_args()

    if (args.seed is not None):
        assert "PYTHONHASHSEED" in os.environ, ("Execution is only deterministic if (next to the -seed parameter) the environmental variable PYTHONHASHSEED is set! Either remove -seed or set PYTHONHASHSEED")
        print(f"Setting seed to {args.seed}")
        # Set inital seeds
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Get a list of seeds for 'num_embeddings' many embeddings
    args.seed = [np.random.randint(4294967296 - 1) for i in range(args.num_embeddings)]

    for dataset in args.dataset:
        graph_name, graph = parse_graph(dataset, args.largest_cc)

        node2vec_graph = Graph(graph, args.p, args.q, not args.no_alias)

        for i in range(args.num_embeddings):
            print("Generating embedding", i)
            embedding = generate(node2vec_graph, args.dimensions, args.walk_length, args.walk_number, seed=args.seed[i])
            if args.output and args.num_embeddings == 1:
                file_name = args.output
            else:
                file_name = os.path.dirname(os.path.abspath(__file__)) + ("/results/node2vec_{0}_{1}.emb").format(graph_name, i)
            print("Save embedding at", file_name)
            save_embedding(embedding, file_name, {
                "algorithm": "node2vec",
                "walk-length": args.walk_length,
                "walk-number": args.walk_number,
                "p": args.p,
                "q": args.q,
                "seed": args.seed[i]
            })
