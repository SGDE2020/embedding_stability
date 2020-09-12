from itertools import islice
import random

import numpy as np
from tqdm import tqdm


class Graph():
    def __init__(self, nx_G, p, q, use_alias=True):
        """
        Class to generate random walks on the given graph as used in node2vec.

        Preprocessing of edge weights (alias drawing) can be turned on or off,
        depending on the particular input graph.
        For very large graphs, alias generation should be turned off, as the
        additional memory usually exeeds the available RAM. It is best turned
        on on medium sized graphs.

        When generating multiple embeddings for the same graph, you should only
        create *one* instance of this class, and call `simulate_walks` multiple
        times. No relevant information is leaked to subsequent calls of that
        function, but it allows the preprocessing to be cached between
        multiple runs.
        """
        self.G = nx_G
        self.p = p
        self.q = q
        self.use_alias = use_alias

        self.alias_edges = {}


    def simulate_walks(self, num_walks, walk_length):
        """
        Performs the given number of walks for every node in the graph. Returns
        a generator that yields the walks in order of nodes, so care should be
        taken as to whether random shuffeling is required in subsequent usage.

        Note that e.g. the word2vec implementation of gensim does not work with
        generators. Collect the result in a list first.
        """
        for node in tqdm(self.G, desc="Generating random walks"):
            for _ in range(num_walks):
                yield self._random_walk(walk_length, node)


    def _random_walk(self, walk_length, start_node):
        walk = [start_node, random.choice(list(self.G[start_node]))]

        while len(walk) < walk_length - 1:
            cur = walk[-1]
            if len(self.G[cur]) > 0:
                prev = walk[-2]

                if self.use_alias:
                    walk.append(self._get_edge_alias(prev, cur, self.G[cur]))
                else:
                    choice = random.choices(list(self.G[cur]), (self._transition_prob(prev, dst_nbr) for dst_nbr in self.G[cur]))[0]
                    walk.append(choice)
            else:
                # no neighbor for this node, thus cannot extend the walk
                break

        # We have to cast the nodes to string here, as the subsequent call to
        # word2vec from gensim does not work well with integers.
        return list(map(str, walk))


    def _get_edge_alias(self, src, dst, neighbors):
        """
        Implements the alias method for more efficient sampling of weighted
        neighbors, see e.g. https://en.wikipedia.org/wiki/Alias_method

        Unlike the reference node2vec implementation, we set up the data for
        each edge just-in-time when needed. Since many pairs of nodes appear
        never after each other in a random walk, this makes 'preprocessing'
        significantly faster and reduces the required memory.
        """
        if not (src, dst) in self.alias_edges:
            probs = np.fromiter((self._transition_prob(src, dst_nbr) for dst_nbr in self.G[dst]), dtype=float)
            norm_const = np.sum(probs)

            self.alias_edges[(src, dst)] = _alias_setup(probs / norm_const)

        return _nth(neighbors, _alias_draw(self.alias_edges[(src, dst)]))


    def _transition_prob(self, src, dst_nbr):
        if dst_nbr == src:
            return 1.0 / self.p
        elif self.G.has_edge(dst_nbr, src):
            return 1.0
        else:
            return 1.0 / self.q


def _alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.min_scalar_type(len(probs)))

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def _alias_draw(alias):
    J = alias[0]
    q = alias[1]

    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

def _nth(iterable, n):
    return next(islice(iterable, n, None))
