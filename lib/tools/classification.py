from enum import Enum

import networkx as nx
import numpy as np


class ClassificationType(Enum):
    MULTILABEL = "multilabel"
    MULTICLASS = "multiclass"


def prepare_classification(graph, node_embeddings=None):
    """
    Performs preparation for a classification task by extracting node labels
    from the given graph into a numpy array.
    Labels are expected to be stored in a node attribute `class`. They will be
    normalized to integers starting from zero if labels are classes.
    In a multilabel classification setting, labels remain untouched.

    Parameters
    ----------
    graph : nx.Graph
        The graph whose nodes should be classified. Labels must be stored in
        the `class` node attributes.
    node_embeddings : np.array
        Optional array of node embeddings. If given, the embedding is used to
        automatically determine if the whole graph or only its largest
        connected component should be considered.

    Returns
    -------
    node_labels : np.array
        Normalized label for each node, ordered as the nodes in the graph.
    distinct_node_labels : int
        Number of unique node labels.
    """
    if node_embeddings is not None and len(graph) != node_embeddings.shape[0]:
        # maybe embedding was generated with largest cc?
        if graph.is_directed():
            graph = max(nx.weakly_connected_component_subgraphs(graph), key=len)
        else:
            graph = max(nx.connected_component_subgraphs(graph), key=len)

        if len(graph) != node_embeddings.shape[0]:
            exit("Embedding size does not fit to given graph (even when only considering largest connected component).")

    example_label = graph.nodes[0]["class"]
    classification_task = determine_classification_task(graph)

    if classification_task == ClassificationType.MULTILABEL:
        node_labels = np.empty((len(graph), example_label.shape[-1]), dtype=example_label.dtype)
        for i, (_, label) in enumerate(graph.nodes(data="class")):
            node_labels[i] = label
        distinct_node_labels = len(np.unique(node_labels, axis=0))
    elif classification_task == ClassificationType.MULTICLASS:
        node_labels = np.empty(len(graph), dtype=int)
        unique_node_labels = dict()
        for i, (_, label) in enumerate(graph.nodes(data="class")):
            node_labels[i] = unique_node_labels.setdefault(label, len(unique_node_labels))
        distinct_node_labels = len(unique_node_labels)

    return node_labels, distinct_node_labels


def determine_classification_task(graph_or_labels):
    """
    Function to determine whether the classification is a multiclass or
    multilabel classification.
    This function works with a graph instance or with a label matrix.
    """
    if isinstance(graph_or_labels, nx.Graph):
        example_label = graph_or_labels.nodes[0]["class"]
        if isinstance(example_label, np.matrix):
            return ClassificationType.MULTILABEL
        elif isinstance(example_label, int) or isinstance(example_label, str):
            return ClassificationType.MULTICLASS
        else:
            raise ValueError("Given graph has an unsupported node label type {}.".format(type(example_label)))
    elif isinstance(graph_or_labels, np.ndarray):
        if graph_or_labels.ndim == 2:
            return ClassificationType.MULTILABEL
        elif graph_or_labels.ndim == 1:
            return ClassificationType.MULTICLASS
        else:
            raise ValueError("Given node label array has unsupported dimensions.")
    else:
        raise ValueError("Could not infer classification task from parameter `graph_or_labels`, type was {}".format(type(graph_or_labels)))


def calc_stable_core(predictions, threshold: float = 0.9):
    predictions = np.array(predictions)
    threshold_count = predictions.shape[0] * threshold

    stable_core = 0
    for node_id in range(predictions.shape[1]):
        _, unique_counts = np.unique(predictions[:,node_id], axis=0, return_counts=True)
        if np.any(unique_counts >= threshold_count):
            stable_core += 1

    return stable_core


def save_classification_results(filename, score_dict, num_classes, splits, repeats):
    """
    Saves classification scores at the given location for later use.

    The file will contain as first line general information the classification,
    including the number of splits and repeats in cross validation.
    Then, each line contains the train and test scores of each run of the
    classifier, seperated by a space.

    Parameters
    ----------
    filename : str
    score_dict : dict with keys "train_score_accuracy", "test_score_accuracy",
        "train_score_f1_micro", "test_score_f1_micro", "train_score_f1_macro",
        "test_score_f1_macro" and "test_predictions"; all except last entry
        should be lists of float, last one a dict
    num_classes : int
        Number of possible classes in case of multiclass classification. Number
        of possible labels in case of multilabel classification.
    splits : int
    repeats : int
    """
    with open(filename, "w") as file:
        print(f"{num_classes} {splits} {repeats}", file=file)
        for data in zip(*score_dict.values()):
            print(" ".join(map(str, data)), file=file)


def load_classification_results(filename, load_predictions=False):
    """
    Reads classification scores from the given location. The file should have
    been generated by `save_classification_results`.

    Parameters
    ----------
    filename : str
    load_predictions : bool

    Returns
    -------
    num_classes : int
        Number of possible classes in case of multiclass classification. Number
        of possible labels in case of multilabel classification.
    splits : int
    repeats : int
    scores : dict with keys "train_score_accuracy", "test_score_accuracy",
        "train_score_f1_micro", "test_score_f1_micro", "train_score_f1_macro",
        "test_score_f1_macro" and "test_predictions"
    """
    with open(filename) as file:
        num_classes, splits, repeats = list(map(int, file.readline().split(" ")))
        total_runs = splits * repeats

        scores = {
            "train_score_accuracy": np.empty(total_runs),
            "test_score_accuracy": np.empty(total_runs),
            "train_score_f1_micro": np.empty(total_runs),
            "test_score_f1_micro": np.empty(total_runs),
            "train_score_f1_macro": np.empty(total_runs),
            "test_score_f1_macro": np.empty(total_runs),
            "test_predictions": list()
        }

        for i, line in enumerate(file):

            split = line.split(" ", maxsplit=6)
            scores["train_score_accuracy"][i] = float(split[0])
            scores["test_score_accuracy"][i] = float(split[1])
            scores["train_score_f1_micro"][i] = float(split[2])
            scores["test_score_f1_micro"][i] = float(split[3])
            scores["train_score_f1_macro"][i] = float(split[4])
            scores["test_score_f1_macro"][i] = float(split[5])


            # numpy inserts newlines into printed arrays, so that we do not
            # have one entry per line anymore.
            # The predictions part may span over multiple lines. Its end can be
            # detected by the occurance of `}`.
            if load_predictions:
                pred_str = split[-1]
                while "}" not in pred_str:
                    pred_str += file.readline().rstrip("\n")
                scores["test_predictions"].append(eval(pred_str.replace(" array(", " np.array(")))
            elif "}" not in split[6]:
                while "}" not in file.readline():
                    pass
    return num_classes, splits, repeats, scores


def load_link_prediction_results(filename, load_predictions=False):
    """
    Reads classification scores from the given location. The file should have
    been generated by `save_classification_results`.

    Parameters
    ----------
    filename : str
    load_predictions : bool

    Returns
    -------
    num_classes : int
        Number of possible classes in case of multiclass classification. Number
        of possible labels in case of multilabel classification.
    splits : int
    repeats : int
    scores : dict with keys "train_score_accuracy", "test_score_accuracy",
        "train_score_f1_micro", "test_score_f1_micro", "train_score_f1_macro",
        "test_score_f1_macro" and "test_predictions"
    """
    with open(filename) as file:
        num_classes, splits, repeats = list(map(int, file.readline().split(" ")))
        total_runs = splits * repeats

        scores = {
            "train_score_accuracy": np.empty(total_runs),
            "test_score_accuracy": np.empty(total_runs),
            "train_score_f1_micro": np.empty(total_runs),
            "test_score_f1_micro": np.empty(total_runs),
            "train_score_f1_macro": np.empty(total_runs),
            "test_score_f1_macro": np.empty(total_runs),
            "train_score_auc": np.empty(total_runs),
            "test_score_auc": np.empty(total_runs),
            "test_predictions": list()
        }

        for i, line in enumerate(file):
            split = line.split(" ", maxsplit=8)

            scores["train_score_accuracy"][i] = float(split[0])
            scores["test_score_accuracy"][i] = float(split[1])
            scores["train_score_f1_micro"][i] = float(split[2])
            scores["test_score_f1_micro"][i] = float(split[3])
            scores["train_score_f1_macro"][i] = float(split[4])
            scores["test_score_f1_macro"][i] = float(split[5])
            scores["train_score_auc"][i] = float(split[6])
            scores["test_score_auc"][i] = float(split[7])

            # numpy inserts newlines into printed arrays, so that we do not
            # have one entry per line anymore.
            # The predictions part may span over multiple lines. Its end can be
            # detected by the occurance of `}`.
            if load_predictions:
                pred_str = split[-1].strip('[]')
                scores["test_predictions"].append(np.fromstring(pred_str, sep=","))

    return num_classes, splits, repeats, scores