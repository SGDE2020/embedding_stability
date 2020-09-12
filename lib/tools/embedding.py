"""Parse parameters lines, read and write embeddings to files
"""
import numpy as np


def read_embedding(file, read_params=False):
    """
    Reads an embedding file. Orders the vectors by id.
    """
    params = parse_param_lines([file.readline(), file.readline()])

    node_embeddings = np.empty((int(params["node_count"]), int(params["embedding_dimension"])))
    lines = [list(map(float, line.split())) for line in file.readlines()]
    node_ids_ordered = sorted([int(line[0]) for line in lines])
    id_to_index_map = dict({node_ids_ordered[i] : i for i in range(len(node_ids_ordered))})
    # min_node_id = min([int(line[0]) for line in lines])
    for line in lines:
        node_embeddings[id_to_index_map[int(line[0])]] = list(map(float, line[1:]))

    if read_params:
        return node_embeddings, params
    return node_embeddings


def save_embedding(embedding, filename, params):
    """Saves an embedding to the given path

    Arguments:
        embedding {np.ndarray or list of lists} -- Calculated embedding
        filename {str} -- Filename of the output file
        params {dict or None} -- Dictionary describing the model parameters.
            These will be written to the first two lines of the embedding.
            node_count and embedding_dimension will be created automatically.
    """
    with open(filename, "w") as f:
        if params is None:
            params = {}

        params["node_count"] = embedding.shape[0]
        params["embedding_dimension"] = embedding.shape[1]

        for line in create_param_lines(params):
            f.write(f"{line}\n")

        lines = []
        for index, data in enumerate(embedding):
            # row consistes of: index [emb_vector]
            line = f"{index} " + " ".join(map(str,data))
            lines.append(line)

        f.write("\n".join(lines))


def parse_param_lines(lines, delimiter=" "):
    """
    Translates two parameter lines to a dict.

    Remember that the values will not be cast to correct types!

    Arguments:
        lines {str[]} -- List of length two of the format returned by create_param_lines.
        Names and values are expected to be separated by delimiter and
        each line must contain a leading comment character (e.g. #)

        delimiter {str} -- The delimiter that was used to separate the names and values

    Returns:
        dict -- Dictionary with params' names as keys and their uncast(!) values.
    """
    assert len(lines) == 2,\
           f"parse_param_lines received not the expected 2, but {len(lines)} entries!"
    # Remove trailing line breaks and leading comment char before zipping to dictionary
    return dict(zip(lines[0][:-1].split(delimiter)[1:], lines[1][:-1].split(delimiter)[1:]))


def create_param_lines(params, delimiter=" "):
    """
    Creates two string lines that describe the used parameters.

    They can be used to be put at the beginning of a file describing the
    parameters used for an embedding or comparison. parse_param_lines can be
    used to translate these params from two lines back to a dict.

    Arguments:
        params {dict} -- Dictionary that holds the params' names as keys and
        the value used as string value (e.g. {"alpha": 0.02})

        delimiter {str} -- Character that the entries should be split on

    Returns:
        str[] -- Array of length two, where the first lines contains the
        param names and the second one the respective values. The entries
        are separated by spaces and there is a leading '#' in each line.
    """
    return [
        "# " + delimiter.join(str(p) for p in list(params.keys())),
        "# " + delimiter.join(str(v) for v in list(params.values()))
    ]


def prepend_param_lines(file_path, param_lines):
    """Prepends the two given param_lines to the specified file and discards the top line

    Arguments:
        file_path {str} -- Path to the file
        param_lines {str[]} -- Two lines that will be added to the beginning of file at file_path
    """
    assert len(param_lines) == 2,\
           f"prepend_param_lines received not the expected 2, but {len(param_lines)} entries!"
    with open(file_path, 'r') as file:
        file.readline() # discard node_count, dimension
        content = file.read()
    with open(file_path, 'w') as file:
        file.write("\n".join([param_lines[0], param_lines[1], content]))

