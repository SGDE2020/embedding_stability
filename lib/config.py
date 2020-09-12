import multiprocessing
import os

# only used for LINE so far
NUM_CORES = multiprocessing.cpu_count()

# directory that contains all the algorithms. I use the directory this project is located in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = BASE_DIR + "/data/"
GRAPH_DIR = BASE_DIR + "/graphs/"

# Embedding Algorithms
BASE_ALGORITHM_DIR = os.path.abspath(BASE_DIR + '/../../')

# source https://github.com/palash1992/GEM
GEM_DIR = BASE_ALGORITHM_DIR + "/GEM/"

# statically compiled COMPILED_LINE (works but may be inefficient, since compilation is not optimised for the system)
LINE_DIR = BASE_DIR + "/line_embedding/COMPILED_LINE/"
