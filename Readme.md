# The Effects of Randomness on the Stability of Node Embeddings

This repository provides the code used in the experiments of our paper

*The Effects of Randomness on the Stability of Node Embeddings*  
*Tobias Schumacher, Hinrikus Wolf, Martin Ritzert, Florian Lemmerich, Jan Bachmann, Florian Frantzen, Max Klabunde, Martin Grohe, and Markus Strohmaier*



Below we provide an overview of all our experiments and how they can be executed.



## Experiments

All Python code (when not explicitly marked other) must be run with python 3.6, since the used tensorflow version 1 is not available in pypi.org for newer python versions.
All used external libraries are with version numbers in `requirements.txt`.

We recommend to run the code within a virtual enviroment and install requirements with in it:
```bash
    python3.6 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
```

### Embedding Generation

Scripts for all our used embedding methods (GraphSAGE, LINE, HOPE, node2vec,
SDNE) are located in the `train/` directory. If you do not want to specify a directory in which the embeddings are saved, there needs to be subdirectory `results/`.
They can be called as follows:

- **GraphSAGE** [[Source]](https://github.com/williamleif/GraphSAGE):

  ```bash
  python3 -m train.graphsage {LOCATION OF GRAPH} -num-embeddings=30  --seed {SEED}
  ```

  The reference implementation linked above is not compatible with newer
  versions of python and networkx. We ported the source code accordingly, but
  made no further changes to the code. The code is located at
  `lib/graphsage_embedding/`.

- **HOPE** [[Source]](https://github.com/palash1992/GEM):

  ```bash
  python3 -m train.hope_generate_embeddings {LOCATION OF GRAPH} --num_embeddings 30 --seed {SEED}
  ```

  An installation of GEM is required. Please see their documentation on how to install GEM.

- **LINE** [[Source]](https://github.com/tangjianpku/LINE):

  ```bash
  python3 -m train.line_generate_embeddings {LOCATION OF GRAPH} --num_embeddings 30 --seed {SEED}
  ```

  A compiled version of LINE in `lib/line_embedding/COMPILED_LINE/` is needed (`line`, `normalize` and `reconstruct`). We changed `line.cpp` to add an argument to seed the initialization of the embedding. Please see their documentation on how to install and compile LINE, but use our modified version. The file can be found in `lib/line_embedding/line.cpp`.

- **node2vec** [[Source]](https://github.com/aditya-grover/node2vec):

  ```bash
  python3 -m train.node2vec {LOCATION OF GRAPH} -num-embeddings=30 -no-alias --seed {SEED}
  ```

  We made changes to the reference implementation, in particular to the random
  walk procedure. The changed code can be found in
  `lib/node2vec_embedding/randomwalk.py`.

- **SDNE** [[Source]](https://github.com/suanrong/SDNE):

  The SDNE reference implementation requires python version `2.7` for execution. For detailed package versions see below.

  ```bash
  python2 -m train.sdne_generate_embeddings -c {LOCATION OF CONFIG FILE}
  ```

  The following package versions are required to properly execute the SDNE implementation:

  | Library | Version |
  |----|----|
  | python | ==2.7 |
  | tensorflow[-gpu] | ==1.12.0 |
  | scikit-learn | ==0.20.4 |
  | configparser | * |

  We slighlty adjusted the config file structure given by the reference implementation as well as the model to fix random seeding. Full details for configuration parameters can be found in the [reference implementation](https://github.com/suanrong/SDNE/tree/master/config). The following changes were made to the configuration structure:
  - `[Output]`:
    - removed `embedding_filename` since the filename for a single embedding is fixed
    - added `num_embeddings` (expecting an `int`) to set the numbers of embeddings that will be created (usually set to `30`)
  - `[Model_Setup]`:
    - `seed` (expecting an `int`) added to fix a random seed (useful for reproducibility)

  Exemplary `Pipfile` (specifying the required package versions) and `.ini` config files are included at `/lib/sdne_embeddings/`.

Embeddings will be stored in plain text in `train/results/` as
`{EMBEDDING ALGORITHM}_{GRAPH NAME}_{ITERATION}.emb`.

In our case, all graphs were stored inside a `graphs/` directory, such that
`{LOCATION OF GRAPH}` was one of the following:

- `barabasi_albert_graphs/ba_N{NUM}_d01.edgelist` with `{NUM}` in 1000, 2000,
  4000, 8000, 16000, 32000, 64000, 128000.
- `barabasi_albert_graphs/ba_N8000_d{DEG}.edgelist` with `{DEG}` in 00025, 0005,
  001, 002, 005, 02, 05, 1.
- `BlogCatalog/`
- `Facebook`
- `Protein/`
- `Cora/`
- `watts_strogatz_graphs/ba_N{NUM}_d01.edgelist` with `{NUM}` in 1000, 2000,
  4000, 8000, 16000, 32000, 64000, 128000.
- `watts_strogatz_graphs/ba_N8000_d{DEG}.edgelist` with `{DEG}` in 00025, 0005,
  001, 002, 005, 02, 05, 1.
- `Wikipedia/`

For reproducibility, an optional seed parameter (integer) can be part of the arguments.
This will set the random number generators from numpy, tensorflow etc. Keep in mind, that for LINE, the seed will only fix the random initialization of the embeddings due to the usage of ASGD (only restricting ourselves to a single thread would eliminate this randomness).

#### Graph Sources

We downloaded the empirical graphs for our study at these locations:

- **BlogCatalog**: http://socialcomputing.asu.edu/datasets/BlogCatalog3
- **Cora**: http://konect.uni-koblenz.de/networks/subelj_cora
- **Facebook**: https://snap.stanford.edu/data/gemsec-Facebook.html (we only used the `government_edges.csv` graph from this dataset)
- **Protein**: https://snap.stanford.edu/node2vec
- **Wikipedia**: https://snap.stanford.edu/node2vec/

### Similarity Tests

All scripts related to our similarity experiments are
located in the corresponding directory `similarity_tests/`.

For some experiments, `graph-tool` is required, which is not automatically
installed when installing the dependencies from our `requirements.txt` file.
Consult their official documentation on how to install `graph-tool` on your
system.
We used version 2.29 in our experiments.

For all experiments, a set of embeddings - corresponding to the graph(s) in the experiment - is required.
We conduct experiments to measure the similarity of two different embeddings on the same graph by the same algorithm.
Our measures (with abbreviations to be used in the similarity tests script) are

- Aligned cosine similarity  `cossim`

- Naive k-nearest-neighbor overlap  `knn`
- Jaccard similarity of k-nearest-neighbor sets `jaccard`
- Ranking Invariance of k-nearest-neighbor sets `ranks`
- Ranked Neighborhood Stability of k-nearest-neighbor sets  `rns`
- k-NN Angle Divergence `angdiv`
- Second order cosine similarity `2ndcos`
- Variants of aligned cosine similarity cosine similarity with respect to different kind of neighbors

To run all experiments and see the results (for all graphs and algorithms), do the following:

1. Use `lib/tools/get_node_info.py` to generate files containing information for nodes of a graph. The graphs have to be available in `graphml` format. Also, a functional installation of `graph-tool` is required.
2. Run the similarity test script (running for all graphs). For a detailed description of the command, see the corresponding section.
3. Use the IPython notebooks to visualize the results:
    - `general_stability_plots.ipynb`: Distribution plots of similarity values.
    - `neighbor_variance_plots_agg.ipynb`: Average mean absolute deviation in degrees for different kind of neighbors.
    - `node_centrality_plots.ipynb`: Similarity measure over node property (e.g. mean aligned cosine similarity over coreness).
    - `synth_plots.ipynb`: Similarity measures over a graph property such as density or size.

Additionally, `tables.ipynb` can be used to generate tables of different similarity measures. The columns are graphs; rows are algorithms. The cell values are aggregated in one of the following ways:

  - Mean of means
  - Standard deviation of means
  - Mean of standard deviations

#### Similarity Test Script

  ```bash
  python3 similarity_tests.py -a {ALGORITHMS} -d {GRAPHS} -t {TESTS} --knn-size {KNN_SIZE} --num-nodes {NUM_NODES} --emb-dir {PATH_TO_EMBEDDINGS} --nodeinfo-dir {PATH_TO_NODEINFO_DIRECTORY} --results-dir {PATH_TO_SAVE_DIRECTORY}
  ```

  where `{ALGORITHMS}` is one or more of `line`, `hope`, `sdne`, `graphsage`, `node2vec`. If the option `-a` is omitted, all algorithms will be used in the experiment.
  `{GRAPHS}` are identifiers of graphs, e.g., `ba_N8000_d1` or `cora`. If the option `-d` is omitted, the experiment will run with default values for the names, that might not match your file names.
  The `-t` option specifies which experiments will be conducted. `{TESTS}` is one or more of `cossim`, `knn`, `jaccard`, `ranks`,  `rns`,  `angdiv`,  `2ndcos`,  `cos`, `orthtf`,  `tfcos`,  `lintf`,  `lincos`. If omitted, all experiments will be run.

The size of the neighborhoods for the nearest-neighbor-based measures can be specified with `--knn-size`.

With `--num-nodes`, the number of nodes of a graph that will be used in the experiment can be specified. Providing a negative value will select all nodes and corresponds to default behavior.
  `{PATH_TO_EMBEDDINGS}`, `{PATH_TO_NODEINFO_DIRECTORY}`, `{PATH_TO_SAVE_DIRECTORY}` are required to run the script. Note that `--results-dir` must have a subdirectory called `procrustes_matrices`.

Additional convenience parameters are documented within the `argparse` of the script.

**Note:** Ranked neighborhood stability `rns`can only be computed if Jaccard similarity `jaccard` and ranking invariance `ranks` for the same value of `--knn-size` have been computed beforehand.

#### Plots

- Plotting of similarity test results is partly integrated with conducting the experiments. All plots are generated in IPython notebooks.
  - `general_stability_plots.ipynb`: Distribution plots of similarity values.
  - `neighbor_variance_plots_agg.ipynb`: Average mean absolute deviation in degrees for different kind of neighbors.
  - `node_centrality_plots.ipynb`: Similarity measure over node property (e.g. mean aligned cosine similarity over coreness).
 -  `synth_plots.ipynb`: Similarity measures over a controlled graph property such as density or size.

### Downstream Classification

All scripts related to our experiments for downstream classification are
located in the corresponding directory `downstream_classification/`.

We tested the performance of AdaBoost, Decision Trees, Neural Networks and
Random Forests on all considered graphs and all generated embeddings.

We made experiments conserning the change in accuracy over different embeddings
for the same graph using the same embedding algorithm and how much predictions
overlap (i) for multiple runs of the classifier and (ii) for runs on different
algorithms.
These experiments can be run as follows:

#### Node Classification

- Accuracy:

  ```bash
  python3 -m downstream_classification.classify {LOCATION OF GRAPH} {LOCATION OF ALL EMBEDDINGS} -model=one_hidden
  ```

- Overlap between multiple runs of the classifier on the same embedding:

  ```bash
  python3 -m downstream_classification.overlap_classifier {LOCATION OF GRAPH} {LOCATION OF EMBEDDING} adaboost decision_tree neural_network random_forest
  ```

  We repeated this experiment for five different embeddings for each graph and
  embedding method.
  We then averaged the results over these five different embeddings.

- Overlap between different embeddings:

  ```bash
  python3 -m downstream_classification.overlap_embeddings {LOCATION OF GRAPH} {EMBEDDINGS} {CLASSIFIER}
  ```

  where `{EMBEDDINGS}` is the list of all embeddings of an embedding algorithm
  for the specified graph.
  `{CLASSIFIER}` can be one of `adaboost`, `decision_tree`, `neuronal_network`
  and `random_forest`.

Results for all downstream tasks are placed in
`downstream_classification/results/`.
For accuracy results, the results are stored in subfolders with the respective
classifier name as dirname.
The file contains as first line information about the task, and then for each
run (cross-validation and repetitions) a line with training and test accuracy.

Results for the other tasks are stored inside the `compare_embedding_errors`
subfolder, for the second task prefixed with `self_`.
The file contains a upper triangle matrix with the pairwise number of nodes
with different predictions.

#### Link Prediction

The experiments are analogue to those on node classification

For Link Prediction, 10% of the edges of graph must be deleted before computing the embeddings.
These deleted edges are later used for traning, together with sampled non-edges.
The Graphs can be reduced via
```bash
    python -m downstream_classification.remove_edges_from_graph {GRAPH_NAME} [--relabel-nodes]
```
The remove takes the larges connected component and only deletes edges such that the resulting graph keeps connected.
The option `--relabel-nodes` is important when the graph is initinally not connect. When set, the nodes will be relabeled with ascending integer numbers, since it is needed for comnputing the stable core.

The corresponding class are:

- Accuracy (AUC)
```bash
    python -m downstream_classification.link_prediction {PATH_TO_REDUCED_GRAPHS} {PATH_TO_EMBEDDING} -classifiers { CLASSIFIER }
```

All individual predictions are saved, so this scipt also generates all needed information to plot stable core


#### Plots

Results for the downstream classification tasks can be visualized in two plots:

- Accuracy results are visualized in a letter plot, code can be found in the
  `plot_scripts/plot_accuracies.py` script.
- Results of both overlap experiments are plotted in one graph, which can be
  generated by executing `downstream_classification/plot_avg.overlap.py`.
  The plots will be saved inside the `plots/` directory.
  Execute the script as follows:
- Link Prediction plots can be created with `plot_scripts/plot_link_prediction.py` and the corresponding table core with `downstream_classification/plot_link_stable_core`

All scripts should be called as modules from the base directory, e.g,

  ```bash
  python3 -m downstream_classification.plot_avg_overlap
  ```



### node2vec case study

The experiment for node2vec can be found within the folder `lib/node2vec_pytorch_zerlegt`

### Additional Comment

If you do not want to run the files from the top-level directory, it may be necessary to add the directory to your `$PYTHONPATH`.
