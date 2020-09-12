import multiprocessing as mp
import os
import pickle
import warnings
from functools import partial
from itertools import combinations
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler, normalize
from tqdm import tqdm

from lib.tools.embedding import parse_param_lines, read_embedding


class Comparison:

    def __init__(self, emb_dir, embeddings):
        """
        emb_dir: str, path where embeddings are saved
        embeddings: list, list of strings that specify the embedding files
        """
        self.dir = emb_dir
        self.embeddings = embeddings
        self.pairs = self._combinations(embeddings)
        self.num_vertices = self.read_vertex_count()

        # use file names without numbering to mark result files
        self.file_prefix = "_".join(self.pairs[0][0].split("_")[:-1])

    def _combinations(self, emb_list):
        """ Computes all different comparison pairs of a list of embeddings """
        return [pair for pair in combinations(emb_list, 2)]

    def _analyse_knn(self, queries, nodes, k, pair):
        """
        This function is called in the multiprocessing of the k-NN overlap.
        """
        indices_0 = np.asarray(queries[pair[0]])
        indices_1 = np.asarray(queries[pair[1]])
        nodes = np.asarray(nodes)
        overlap = {}
        for i in range(len(nodes)):
            overlap[nodes[i]] = len(
                np.intersect1d(indices_0[i, 1:(k + 1)], indices_1[i, 1:(k + 1)], assume_unique=True)) / k
        return list(overlap.values())

    def _analyse_jaccard(self, queries, nodes, k, pair):
        """This function is called in the multiprocessing of jaccard score"""
        indices_0 = np.asarray(queries[pair[0]])
        indices_1 = np.asarray(queries[pair[1]])
        nodes = np.asarray(nodes)
        jaccard_score = {}
        for i in range(len(nodes)):
            jaccard_score[nodes[i]] = \
                len(np.intersect1d(indices_0[i, 1:(k + 1)], indices_1[i, 1:(k + 1)], assume_unique=True)) / \
                len(np.union1d(indices_0[i, 1:(k + 1)], indices_1[i, 1:(k + 1)]))
        return list(jaccard_score.values())

    def _analyse_ranks(self, queries, nodes, k, pair):
        """
        This function is called in the multiprocessing of the k-NN ranking invariance.
        """
        indices_0 = np.asarray(queries[pair[0]])
        indices_1 = np.asarray(queries[pair[1]])
        nodes = np.asarray(nodes)
        invariance_scores = {}
        s_max = sum(1.0 / i for i in range(1, k + 1))
        for i in range(len(nodes)):
            vals0, indices0 = np.unique(indices_0[i, 1:(k + 1)], return_index=True)
            vals1, indices1 = np.unique(indices_1[i, 1:(k + 1)], return_index=True)
            ranks0 = dict(zip(vals0, indices0 + 1))
            ranks1 = dict(zip(vals1, indices1 + 1))
            overlap = np.intersect1d(vals0, vals1, assume_unique=True)
            res = 0.0
            for u in overlap:
                res += 2 / (1 + abs(ranks0[u] - ranks1[u])) / (ranks0[u] + ranks1[u])
            invariance_scores[nodes[i]] = res / s_max

        return list(invariance_scores.values())

    def _analyse_angle_divergence(self, queries, normed_embs, nodes, k, pair):
        """
        This function is called in the multiprocessing of the k-NN angle divergence.
        """

        # Convert the indices of nearest neighbors back into numpy
        indices_0 = np.asarray(queries[pair[0]])
        indices_1 = np.asarray(queries[pair[1]])

        # Convert the embeddings and nodes back into numpy
        norm_emb_0 = np.asarray(normed_embs[pair[0]])
        norm_emb_1 = np.asarray(normed_embs[pair[1]])
        nodes = np.asarray(nodes)

        # Compute the second order cosine similarity
        pair_results = []
        for i in range(len(nodes)):
            # Build the set of nearest neighbors w.r.t. both embeddings
            # Use indices from 1 to k+1, because the first entry will always be the node itself
            neighbors_union = np.union1d(indices_0[i, 1:(k + 1)], indices_1[i, 1:(k + 1)])

            # Vectors of cosine similarity values of nearest neighbors
            cossim_vec0 = np.squeeze(cos_sim(norm_emb_0[neighbors_union], norm_emb_0[nodes[i]].reshape(1, -1)))
            cossim_vec1 = np.squeeze(cos_sim(norm_emb_1[neighbors_union], norm_emb_1[nodes[i]].reshape(1, -1)))

            # clip cossim values to feasible interval, which it might leave due to numerical issues
            cossim_vec0 = np.clip(cossim_vec0, a_min=-1, a_max=1)
            cossim_vec1 = np.clip(cossim_vec1, a_min=-1, a_max=1)

            # convert to degrees
            m0 = np.degrees(np.arccos(cossim_vec0))
            m1 = np.degrees(np.arccos(cossim_vec1))

            # Cosine similarity between similarity vectors
            pair_results.append(np.mean(np.abs(m0 - m1)))
        return pair_results

    def _analyse_second_cossim(self, queries, normed_embs, nodes, k, pair):
        """
        This function is called in the multiprocessing of the second order cosine similarity.
        """

        # Convert the indices of nearest neighbors back into numpy
        indices_0 = np.asarray(queries[pair[0]])
        indices_1 = np.asarray(queries[pair[1]])

        # Convert the embeddings and nodes back into numpy
        norm_emb_0 = np.asarray(normed_embs[pair[0]])
        norm_emb_1 = np.asarray(normed_embs[pair[1]])
        nodes = np.asarray(nodes)

        # Compute the second order cosine similarity
        pair_results = []
        for i in range(len(nodes)):
            # Build the set of nearest neighbors w.r.t. both embeddings
            # Use indices from 1 to k+1, because the first entry will always be the node itself
            neighbors_union = np.union1d(indices_0[i, 1:(k + 1)], indices_1[i, 1:(k + 1)])

            # Vectors of cosine similarity values of nearest neighbors
            m0 = cos_sim(norm_emb_0[neighbors_union], norm_emb_0[nodes[i]].reshape(1, -1))
            m1 = cos_sim(norm_emb_0[neighbors_union], norm_emb_0[nodes[i]].reshape(1, -1))

            # Cosine similarity between similarity vectors
            pair_results.append(float(np.dot(m0, m1) / (np.linalg.norm(m0) * np.linalg.norm(m1))))
        return pair_results

    def k_nearest_neighbors(self, nodes=None, append=False, samples=100, k=10, jaccard=False, load=False,
                            kload_size=100, save=True, save_path=None, num_processes=4):
        """
        Computes the k nearest neighbors to some specified nodes with respect to cosine similarity. As an intermediate
        step, it computes the 100 nearest neighbors and saves them to file. Based on these neighbors, the k-nn overlaps
        are computed.

        Args:
            nodes: list, that specifies the nodes
            append: bool, if nodes are specified, whether additional nodes will be sampled and added to nodes
            samples: int, number of nodes that will be sampled
            k: int, number of neighbors that will be used in the comparison
            jaccard: bool, whether jaccard score or overlap will be used as similarity measure
            load: bool, whether a file of computed neighbors will be used
            kload_size: Size of k in knn file to load
            save: bool, whether the results should be saved in a text file
            save_path: str, path where the file will be saved
            num_processes: number of random processes in parallelization

        Returns:
            dict, "nodes": array of nodes, "overlaps": array of overlaps, columns are values per node; or array of
            jaccard scores
        """
        # Handle the nodes input: Whether to sample nodes, use given nodes, or both.
        nodes = self._get_nodes(nodes, samples, append)

        # Load k nearest neighbors to save time? Else we will have to compute them first.
        if load is True:
            file_name = self.file_prefix + "_" + str(kload_size) + "nns.pickle"
            with open(os.path.join(save_path, file_name), "rb") as pickle_file:
                queries = pickle.load(pickle_file)
            assert list(queries.keys()) == self.embeddings, ("Keys of loaded queries do not match"
                                                             " with available embeddings")

        # Normalizing the embeddings to be able to use distance as a proxy for cosine similarity
        # BallTree from sklearn is used to compute the neighbors efficiently
        else:
            queries = self.nearest_neighbors_queries(nodes, k, save_path)
        # Store the naive overlaps for all pairs

        print(f"\n\n {queries.keys()} \n")

        # Use multiprocessing to speed up overlap computation.
        # Too much data is passed to the processes which makes it inefficient.
        # Possibly, it is faster to store the data as a file as an intermediate step.
        # only run if less than 100 neighbors are queried, as too large neighborhoods may cause memory issues when
        # distributing tasks
        if num_processes > 1 and k <= 100:
            with mp.Pool(num_processes) as p:
                # arguments passed in multiprocessing must be pickable
                p_queries = queries
                p_nodes = nodes.tolist()
                if jaccard:
                    multiprocess_func = partial(self._analyse_jaccard, p_queries, p_nodes, k)
                else:
                    multiprocess_func = partial(self._analyse_knn, p_queries, p_nodes, k)
                li_overlap = p.map(multiprocess_func, self.pairs)
        else:
            if jaccard:
                li_overlap = [self._analyse_jaccard(queries, nodes.tolist(), k, pair) for pair in self.pairs]
            else:
                li_overlap = [self._analyse_knn(queries, nodes.tolist(), k, pair) for pair in self.pairs]

        # Convert the result into numpy
        overlap = np.asarray(li_overlap)

        # Save the results
        if jaccard:
            nodes_suffix = "jaccard_nodes"
            scores_suffix = f"{k}nn_jaccard"
        else:
            nodes_suffix = "overlap_nodes"
            scores_suffix = f"{k}nn_overlap"
        if save is True:
            np.save(os.path.join(save_path, f"{self.file_prefix}_{nodes_suffix}"), nodes)
            np.save(os.path.join(save_path, f"{self.file_prefix}_{scores_suffix}"), overlap)
        return {"nodes": nodes, "overlaps": overlap}

    def jaccard_similarity(self, nodes=None, append=False, samples=100, k=10, load=False, kload_size=100, save=True,
                           save_path=None, num_processes=4):
        """
        Alias for k_nearest_neighbors with jaccard=True.
        See k_nearest_neighbors for detailed documentation.
        """
        return self.k_nearest_neighbors(nodes=nodes, append=append, samples=samples, k=k, jaccard=True, load=load,
                                        kload_size=kload_size, save=save, save_path=save_path,
                                        num_processes=num_processes)

    def ranking_invariance(self, nodes=None, append=False, samples=100, k=10, load=False, kload_size=100, save=True,
                           save_path=None, num_processes=4):
        """
        Computes the invariance of the ranks of the k nearest neighbors with respect to cosine similarity. As an
        intermediate step, it computes the k nearest neighbors and saves them to file. Based on these neighbors,
        the rank invariance scores are computed.

        Args:
            nodes: list, that specifies the nodes
            append: bool, if nodes are specified, whether additional nodes will be sampled and added to nodes
            samples: int, number of nodes that will be sampled
            k: int, number of neighbors that will be used in the comparison
            load: bool, whether a file of computed neighbors should be loaded instead of querying neighbors
            kload_size: Size of k in knn file to load
            save: bool, whether the results should be saved in a text file
            save_path: str, path where the file will be saved
            num_processes: number of random processes in parallelization

        Returns:
            dict, "nodes": array of nodes, "scoress": array of scoress, columns are values per node;
            or array of jaccard scores
        """
        # Handle the nodes input: Whether to sample nodes, use given nodes, or both.
        nodes = self._get_nodes(nodes, samples, append)

        # Load 100 nearest neighbors to save time? Else we will have to compute them first.
        if load is True:
            file_name = self.file_prefix + "_" + str(kload_size) + "nns.pickle"
            with open(os.path.join(save_path, file_name), "rb") as pickle_file:
                queries = pickle.load(pickle_file)
            assert list(queries.keys()) == self.embeddings, ("Keys of loaded queries do not match"
                                                             " with available embeddings")

        # Normalizing the embeddings to be able to use distance as a proxy for cosine similarity
        # BallTree from sklearn is used to compute the neighbors efficiently
        else:
            queries = self.nearest_neighbors_queries(nodes, k, save_path)

        print(f"\n\n {queries.keys()} \n")

        # Use multiprocessing to speed up overlap computation.
        # Too much data is passed to the processes which makes it inefficient.
        # Possibly, it is faster to store the data as a file as an intermediate step.
        # only run if less than 100 neighbors are queried, as too large neighborhoods may cause memory issues when
        # distributing tasks
        if num_processes > 1 and k <= 100:
            with mp.Pool(num_processes) as p:
                # arguments passed in multiprocessing must be pickable
                multiprocess_func = partial(self._analyse_ranks, queries, nodes.tolist(), k)
                li_scores = p.map(multiprocess_func, self.pairs)
        else:
            li_scores = [self._analyse_ranks(queries, nodes.tolist(), k, pair) for pair in self.pairs]
        # Convert the result into numpy
        scores = np.asarray(li_scores)

        # Save the results
        nodes_suffix = "ranks_nodes"
        scores_suffix = f"{k}nn_ranks"

        if save is True:
            np.save(os.path.join(save_path, f"{self.file_prefix}_{nodes_suffix}"), nodes)
            np.save(os.path.join(save_path, f"{self.file_prefix}_{scores_suffix}"), scores)
        return {"nodes": nodes, "scores": scores}

    def ranked_neighborhood_stability(self, k=10, save=True, save_path=None):
        """
        Computes the node-wise ranked neighborhood stability scores as defined by Wang et al. Requires that
        corresponding Jaccard similarities and ranking invariance for the exact given k's have been computed beforehand,
        as it only loads the corresponding matrices and multiplies them element-wise.

        Args:
            k: int, number of neighbors that will be used in the comparison
            save: bool, whether the results should be saved in a text file
            save_path: str, path where the file will be saved

        Returns:
            array of scores, columns are values per node; or array of ranked neighborhood stability scores
        """

        # Load correct rank invariance and Jaccard similarity files
        jaccard_suffix = f"{k}nn_jaccard"
        ranks_suffix = f"{k}nn_ranks"

        jaccard_mat = np.load(os.path.join(save_path, f"{self.file_prefix}_{jaccard_suffix}.npy"))
        ranks_mat = np.load(os.path.join(save_path, f"{self.file_prefix}_{ranks_suffix}.npy"))

        # multiply matrices element-wise to obtain ranked neighborhood similarity scores
        scores = np.multiply(jaccard_mat, ranks_mat)

        # save the results
        if save is True:
            np.save(os.path.join(save_path, f"{self.file_prefix}_rns"), scores)

        return scores

    def knn_angle_divergence(self, nodes=None, append=False, num_samples=1000, k=10, load=True, kload_size=100,
                             save=False, save_path=None, num_processes=4):
        """ Computes second order angle divergence.

        Args:
            nodes: list, that specifies the nodes
            append: bool, if nodes are specified, whether additional nodes will be sampled and added to nodes
            num_samples: int, number of nodes that will be sampled
            k: int, number of neighbors that will be used in the comparison
            load: bool, whether to load nearest neighbors from file
            kload_size: Size of k in knn file to load
            save: bool, whether the results should be saved
            save_path: str, path where the file will be saved
            num_processes: number of random processes in parallelization

        Returns:
            nodes: numpy array of used nodes
            results: numpy array of similarity values of size (number of embedding pairs, number of nodes)
        """

        # Handle the nodes input: Whether to sample nodes, use given nodes, or both.
        nodes = self._get_nodes(nodes, num_samples, append)

        # Load required data: nearest neighbors, embeddings
        normed_embs = {}

        if load:
            # Load nearest neighbors from file
            file_name = self.file_prefix + "_" + str(kload_size) + "nns.pickle"
            with open(os.path.join(save_path, file_name), "rb") as pickle_file:
                queries = pickle.load(pickle_file)
            assert list(queries.keys()) == self.embeddings, ("Keys of loaded queries do not match"
                                                             " with available embeddings")
            for emb in tqdm(self.embeddings, desc="Loading nearest neighbor files"):
                normed_embs[emb] = normalize(self.read_embedding(os.path.join(self.dir, emb)), norm='l2', copy=False)
        else:
            queries, normed_embs = self.nearest_neighbors_queries(nodes, k, save_path, return_embeddings=True)

        # Use multiprocessing to speed up overlap computation.
        # Too much data is passed to the processes which makes it inefficient.
        # Possibly, it is faster to store the data as a file as an intermediate step.
        # only run if less than 100 neighbors are queried, as too large neighborhoods may cause memory issues when
        # distributing tasks

        # arguments passed in multiprocessing must be pickable
        p_normed_embs = dict([(key, norm_emb.tolist()) for key, norm_emb in normed_embs.items()])

        if num_processes > 1 and k <= 100:
            with mp.Pool(num_processes) as p:
                li_results = p.map(partial(self._analyse_angle_divergence, queries, p_normed_embs, nodes.tolist(), k),
                                   self.pairs)
        else:
            li_results = [self._analyse_angle_divergence(queries, p_normed_embs, nodes.tolist(), k, pair)
                          for pair in self.pairs]

        results = np.asarray(li_results)

        if save is True:
            np.save(os.path.join(save_path, f"{self.file_prefix}_{k}nn_angdiv"), results)

        return nodes, results

    def second_order_cosine_similarity(self, nodes=None, append=False, num_samples=1000, k=10, load=True, save=False,
                                       save_path=None, num_processes=4):
        """ Computes second order cosine similarity.

        Args:
            nodes: list, that specifies the nodes
            append: bool, if nodes are specified, whether additional nodes will be sampled and added to nodes
            num_samples: int, number of nodes that will be sampled
            k: int, number of neighbors that will be used in the comparison
            load: bool, whether to load nearest neighbors from file
            save: bool, whether the results should be saved
            save_path: str, path where the file will be saved
            num_processes: number of random processes in parallelization

        Returns:
            nodes: numpy array of used nodes
            results: numpy array of similarity values of size (number of embedding pairs, number of nodes)
        """

        # Handle the nodes input: Whether to sample nodes, use given nodes, or both.
        nodes = self._get_nodes(nodes, num_samples, append)

        # Load required data: nearest neighbors, embeddings
        normed_embs = {}

        if load:
            # Load nearest neighbors from file
            file_name = self.file_prefix + "_100nns.pickle"
            with open(os.path.join(save_path, file_name), "rb") as pickle_file:
                queries = pickle.load(pickle_file)
            assert list(queries.keys()) == self.embeddings, ("Keys of loaded queries do not match with "
                                                             "available embeddings")
            for emb in tqdm(self.embeddings, desc="Loading nearest neighbor files"):
                normed_embs[emb] = normalize(self.read_embedding(os.path.join(self.dir, emb)), norm='l2', copy=False)
        else:
            queries, normed_embs = self.nearest_neighbors_queries(nodes, k, save_path, return_embeddings=True)

        # Start computation of second order cosine similarity
        # arguments passed in multiprocessing must be pickable
        p_normed_embs = dict([(key, norm_emb.tolist()) for key, norm_emb in normed_embs.items()])
        p_nodes = nodes.tolist()
        if num_processes > 1 and k <= 100:
            with mp.Pool(num_processes) as p:
                li_results = p.map(partial(self._analyse_second_cossim, queries, p_normed_embs, p_nodes, k), self.pairs)
        else:
            li_results = [self._analyse_second_cossim(queries, p_normed_embs, p_nodes, k, pair) for pair in self.pairs]

        results = np.asarray(li_results)

        if save is True:
            np.save(os.path.join(save_path, f"{self.file_prefix}_{k}nn_2nd_order_cossim"), results)

        return nodes, results

    def nearest_neighbors_queries(self, nodes, k, save_path, return_embeddings=False):
        """Uses a ball tree to compute the nearest neighbors in the embedding space"""
        queries = {}
        normed_embs = {}
        # Normalizing the embeddings to be able to use distance as a proxy for cosine similarity
        # BallTree from sklearn is used to compute the neighbors efficiently
        if return_embeddings:
            for emb in tqdm(self.embeddings, desc="Querying nearest neighbors"):
                normed_embs[emb] = normalize(self.read_embedding(os.path.join(self.dir, emb)), norm='l2', copy=False)
                ball_tree = BallTree(normed_embs[emb], leaf_size=40)
                queries[emb] = ball_tree.query(normed_embs[emb][nodes, :], k=k + 1, return_distance=False).tolist()
        else:
            for emb in tqdm(self.embeddings, desc="Querying nearest neighbors"):
                normalized_embedding = normalize(self.read_embedding(os.path.join(self.dir, emb)), norm='l2',
                                                 copy=False)
                ball_tree = BallTree(normalized_embedding, leaf_size=40)
                # Query the k+1 nearest neighbors, because a node will always be the closest neighbor to itself
                queries[emb] = ball_tree.query(normalized_embedding[nodes, :], k=k + 1, return_distance=False).tolist()

        # Save the computed neighbors to be able to skip the computation
        self.save_pickle(queries, save_path, self.file_prefix + "_" + str(k) + "nns")

        if return_embeddings:
            return queries, normed_embs
        else:
            return queries

    def sample_nodes(self, k):
        """
        Sample unique nodes of an embedding

        Args:
            k: int, number of nodes to sample

        Returns:
            numpy array of node ids of length k if k is smaller than the number of nodes available.
            Otherwise, all available nodes are returned.
        """
        vertices = np.arange(self.num_vertices)
        np.random.shuffle(vertices)
        return vertices[:min(k, self.num_vertices)]

    def cosine_similarity(self, nodes=None, append=False, num_samples=1000, save=False, save_path=None):
        """
        Computes the cosine similarity between some specified nodes. The similarity matrices are used in comparisons
        of different embeddings.

        Args:
            nodes: list, that specifies the nodes
            append: bool, if nodes are specified, whether additional nodes will be sampled and added to nodes
            num_samples: int, number of nodes that will be sampled
            save: bool, whether the results should be saved
            save_path: str, path where the file will be saved

        Returns:
            dict, "nodes": numpy array of nodes, "diffs": list of pairwise differences,
                "sims": dict of similarity matrices (list of lists) for every embedding
        """

        # Handle the nodes input: Whether to sample nodes, use given nodes, or both.
        nodes = self._get_nodes(nodes, num_samples, append)

        # Calculate cosine similarity matrices for all embeddings
        similarities = {}
        for emb in tqdm(self.embeddings, desc="Computing similarity matrices"):
            # Using linear kernel is equivalent to cosine similarity on l2-normed data, but faster.
            # By normalizing in the beginning, less norms have to be computed
            normed_emb = normalize(self.read_embedding(os.path.join(self.dir, emb)), norm='l2', copy=False)
            similarities[emb] = linear_kernel(normed_emb[nodes, :])

        # Compute similarity between pairs of embeddings. The measure is:
        # Frobenius norm of the difference of similarity matrices * 1000 / (number of elements in matrix)
        differences = []
        for pair in tqdm(self.pairs, desc="Computing pairwise similarity"):
            dif = np.linalg.norm(similarities[pair[0]] - similarities[pair[1]], ord='fro') * 1000 / (
                        len(nodes) * len(nodes))
            differences.append(dif)

        if save is True:
            self.save_pickle(dict([(key, sim.tolist()) for key, sim in similarities.items()]), save_path,
                             self.file_prefix + "_cossims")

        return {"nodes": nodes, "diffs": differences, "sims": similarities}

    def procrustes_transformation(self, center=False, save_path=None):
        """ Performs orthogonal transformation (Procrustes problem) between two embeddings and saves transformation
        matrices as well as vector of resulting errors
        """
        # TODO: replace with scipy.spatial.procrustes and scipy.spatial.orthogonal_procrustes?

        # Check directory structure
        if save_path is None:
            save_path = os.getcwd()
        matrix_path = os.path.join(save_path, "procrustes_matrices")
        if os.path.exists(matrix_path) is False:
            try:
                os.mkdir(matrix_path)
            except OSError:
                print(f"Creation of directory failed. ({matrix_path})")

        # Set up file naming
        if center is True:
            matrix_name = "linQMatrix"
            results_suffix = "linproc_errors"
        else:
            matrix_name = "QMatrix"
            results_suffix = "procrustes_errors"

        # Do the transformation
        normed_embs = {}
        if center:
            for emb in tqdm(self.embeddings, desc="Reading embeddings"):
                scaler = StandardScaler()
                normed_embs[emb] = scaler.fit_transform(
                    self.read_embedding(os.path.join(self.dir, emb))[np.arange(self.num_vertices)])
        else:
            for emb in tqdm(self.embeddings, desc="Reading embeddings"):
                normed_embs[emb] = normalize(
                    self.read_embedding(os.path.join(self.dir, emb))[np.arange(self.num_vertices)], norm='l2',
                    copy=False)

        results = []
        for pair in tqdm(self.pairs, desc="Aligning embeddings"):
            # compute transformation matrix Q such that W2Q ~= W1
            W1, W2 = np.array(normed_embs[pair[0]]), np.array(normed_embs[pair[1]])

            # use well-established SVD solution
            U, _, VT = np.linalg.svd(W2.T.dot(W1))
            Q = U.dot(VT)

            # save transformation matrix for possible later use
            emb_number_0 = pair[0].split("_")[-1][:-4]
            emb_number_1 = pair[1].split("_")[-1][:-4]
            np.save(os.path.join(matrix_path, f"{self.file_prefix}_{matrix_name}_{emb_number_0}_{emb_number_1}"), Q)

            # append error of procrustes transformation to results
            results.append(np.linalg.norm(W2.dot(Q) - W1))

        np.save(os.path.join(save_path, f"{self.file_prefix}_{results_suffix}"), results)
        return results

    def cossim_analysis(self, center=False, nodes=None, save_path=None):
        """ Computes aligned cosine similarity values. Internally performs orthogonal transformation (Procrustes
        problem) between two embeddings and saves transformation matrices as well as vector of resulting errors
        """

        # Check the directory structure
        if save_path is None:
            save_path = os.getcwd()
        matrix_path = os.path.join(save_path, "procrustes_matrices")
        if os.path.exists(matrix_path) is False:
            raise ValueError(f"Matrix path does not exist ({matrix_path})")

        # Warn about deprecated arguments
        if nodes is not None:
            warnings.warn("nodes argument is deprecated. All nodes are used instead.", DeprecationWarning)

        # Set up file naming
        if center is True:
            matrix_name = "linQMatrix"
            results_suffix = "linproc_cossim"
        else:
            matrix_name = "QMatrix"
            results_suffix = "aligned_cossim"

        # Read the embeddings
        normed_embs = {}
        if center:
            for emb in tqdm(self.embeddings, desc="Reading embeddings"):
                scaler = StandardScaler()
                normed_embs[emb] = scaler.fit_transform(
                    self.read_embedding(os.path.join(self.dir, emb))[np.arange(self.num_vertices)])
        else:
            for emb in tqdm(self.embeddings, desc="Reading embeddings"):
                normed_embs[emb] = normalize(
                    self.read_embedding(os.path.join(self.dir, emb))[np.arange(self.num_vertices)], norm='l2',
                    copy=False)

        # Do the analysis
        emb_ind = -1
        results = []
        for pair in tqdm(self.pairs, desc="Comparing embeddings"):

            # only update first embedding if it does not change
            if emb_ind != pair[0]:
                emb_ind = pair[0]

            W1 = normed_embs[emb_ind]

            # transform W2 into W1 -> load matrix from previous step
            emb_number_0 = pair[0].split("_")[-1][:-4]
            emb_number_1 = pair[1].split("_")[-1][:-4]
            Q = np.load(
                os.path.join(matrix_path, f"{self.file_prefix}_{matrix_name}_{emb_number_0}_{emb_number_1}.npy"))
            W2 = normed_embs[pair[1]].dot(Q)

            pair_results = np.array([cosine(W1[i], W2[i]) for i in range(self.num_vertices)])
            results.append(pair_results)

        results = np.asarray(results)
        np.save(os.path.join(save_path, f"{self.file_prefix}_{results_suffix}"), results)
        return np.arange(self.num_vertices), results

    def linproc_transformation(self, save_path=None):
        """ Performs orthogonal transformation (Procrustes problem) between two embeddings and saves transformation
        matrices as well as vector of resulting errors
        """
        warnings.warn(
            "linproc_transformation is deprecated. Use procrustes_transformation with center=True instead.",
            DeprecationWarning)
        return self.procrustes_transformation(center=True, save_path=save_path)

    def linproc_analysis(self, nodes=None, save_path=None):
        """ Performs orthogonal transformation (Procrustes problem) between two embeddings and saves transformation
        matrices as well as vector of resulting errors
        """
        warnings.warn(
            "linproc_analysis is deprecated. Use procrustes_analysis with center=True instead.",
            DeprecationWarning)
        return self.cossim_analysis(center=True, nodes=nodes, save_path=save_path)

    def _get_nodes(self, nodes, num_samples, append):
        """
        Handles getting nodes for the experiments.

        Args:
            nodes: list, node ids
            num_samples: int, how many nodes should be sampled
            append: bool, whether to append sampled nodes to specified nodes

        Returns:
            numpy array of (sampled) node ids
        """
        if nodes is None:
            nodes = self.sample_nodes(num_samples)
        elif append is True:
            # allows specified nodes to be taken twice
            nodes.extend(self.sample_nodes(num_samples))
        return np.asarray(nodes)

    def save_pickle(self, obj, save_path, file_name):
        if save_path is None:
            save_path = os.getcwd()
        if file_name is None:
            # generate name of report from embedding input: use name of first embedding file without number information
            file_name = self.file_prefix
        with open(os.path.join(save_path, f"{file_name}.pickle"), "wb") as f:
            pickle.dump(obj, f)

    def get_combinations(self):
        return self.pairs

    def get_vertex_count(self):
        return self.num_vertices

    def read_vertex_count(self):
        """ First lines of .emb files contains vertex count. """
        if self.pairs is not None:
            with open(os.path.join(self.dir, self.pairs[0][0]), "r") as f:
                param_lines = [f.readline(), f.readline()]
                return int(parse_param_lines(param_lines)['node_count'])
        else:
            print("No embedding files found.")

    def read_embedding(self, path):
        with open(path, "r") as f:
            return read_embedding(f)
