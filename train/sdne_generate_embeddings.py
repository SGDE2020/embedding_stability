'''
Reference implementation of SDNE

Author: Xuanrong Yao, Daixin wang

Slight modifications were introduced to better match
the ecosystem (especially repetition of embedding creation,
fixation of random seeds and I/O formatting).

Contrary to all other training algorithms, Python 2.7 is
required to run the SDNE embedding algorithm.
An exemplary configuration file can be seen in
`lib/sdne_embeddings/example_config.ini`.

for more detail, refer to the paper:
SDNE: structural deep network embedding
Wang, Daixin and Cui, Peng and Zhu, Wenwu
Knowledge Discovery and Data Mining (KDD), 2016
'''

#!/usr/bin/python2
# -*- coding: utf-8 -*-
from lib.sdne_embeddings.graph import Graph
from lib.sdne_embeddings.config import Config
from lib.sdne_embeddings.model.sdne import SDNE
from lib.sdne_embeddings.utils.utils import *

import scipy.io as sio
import numpy as np
import time
import copy
from optparse import OptionParser
import os

# Python 2.7 copy of lib.tools.embedding.create_param_lines
def _create_param_lines(params, delimiter=" "):
    """Creates two string lines to write embedding information to the start of an .emb file.

    This is a direct copy of lib.tools.embedding.create_param_lines to use with Python 2.7.

    Arguments:
        params {dict} -- Parameters to write

    Keyword Arguments:
        delimiter {str} -- Separator used between the parameters (default: {" "})

    Returns:
        list -- Two strings in a list, where the first one contains the key names and the second one the respective values.
    """
    return [
        "# " + delimiter.join(str(p) for p in list(params.keys())),
        "# " + delimiter.join(str(v) for v in list(params.values()))
    ]

# Python 2.7 copy of lib.tools.embedding.save_embedding
def _write_df_to_emb(embedding, file_name, params=None):
    """Write an embedding to a file.

    Arguments:
        embedding {np.ndarray} -- Created embedding
        file_name {str} -- Path to the file that will be created

    Keyword Arguments:
        params {dict} -- Params to be written to the first two lines of the file.
            node_count and embedding_dimension will be written by default (default: {None})
    """
    with open(file_name, 'w') as f:
        if params is None:
            params = {}
        params["node_count"] = np.size(embedding, axis=0)
        params["embedding_dimension"] = np.size(embedding, axis=1)
        for line in _create_param_lines(params):
            f.write("{0}\n".format(line))

        lines = []
        for index, data in enumerate(embedding):
            #row consistes of: index [emb_vector]
            line = "{0} ".format(index) + " ".join(map(str,data))
            lines.append(line)

        f.write("\n".join(lines))

if __name__ == "__main__":
    # Gather configuration
    parser = OptionParser()
    parser.add_option("-c",dest = "config_file", action = "store", metavar = "CONFIG FILE")
    options, _ = parser.parse_args()
    if options.config_file == None:
        raise IOError("no config file specified")

    config = Config(options.config_file)

    # Build graph from file
    train_graph_data = Graph(config.train_graph_file, config.ng_sample_ratio)

    if config.origin_graph_file:
        origin_graph_data = Graph(config.origin_graph_file, config.ng_sample_ratio)

    if config.label_file:
        #load label for classification
        train_graph_data.load_label_data(config.label_file)

    config.struct[0] = train_graph_data.N

    # Set inital seeds
    np.random.seed(config.seed)
    # Get a list of seeds for 'num_embeddings' many embeddings
    seeds = [np.random.randint(4294967296 - 1) for i in range(config.num_embeddings)]


    for i in range(config.num_embeddings):
        print "Creating embedding {0} of {1}".format(i+1, config.num_embeddings)
        model = SDNE(config, seeds[i])
        model.do_variables_init(train_graph_data)
        embedding = None
        while (True):
            mini_batch = train_graph_data.sample(config.batch_size, do_shuffle = False)
            if embedding is None:
                embedding = model.get_embedding(mini_batch)
            else:
                embedding = np.vstack((embedding, model.get_embedding(mini_batch)))
            if train_graph_data.is_epoch_end:
                break

        epochs = 0
        batch_n = 0

        tt = time.ctime().replace(' ','-')
        print "!!!!!!!!!!!!!"
        while (True):
            if train_graph_data.is_epoch_end:
                loss = 0
                if epochs % config.display == 0:
                    embedding = None
                    while (True):
                        mini_batch = train_graph_data.sample(config.batch_size, do_shuffle = False)
                        loss += model.get_loss(mini_batch)
                        if embedding is None:
                            embedding = model.get_embedding(mini_batch)
                        else:
                            embedding = np.vstack((embedding, model.get_embedding(mini_batch)))
                        if train_graph_data.is_epoch_end:
                            break

                    print "Epoch : %d loss : %.3f" % (epochs, loss)
                if epochs == config.epochs_limit:
                    print "exceed epochs limit terminating"
                    break
                epochs += 1
            mini_batch = train_graph_data.sample(config.batch_size)
            loss = model.fit(mini_batch)
        file_path = os.path.dirname(os.path.abspath(__file__)) + ("/results/sdne_{0}_{1}.emb").format(config.graph_name, i)
        print "Writing embedding to {0}".format(file_path)
        _write_df_to_emb(embedding, file_path, {
                "algorithm": "sdne",
                "implementation": "suanrong",
                "layers": "-".join([str(l) for l in config.struct]),
                "beta": config.beta,
                "alpha": config.alpha,
                "gamma": config.gamma,
                "L2-param": config.reg,
                "epochs": config.epochs_limit,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate
            })
        model.close()
