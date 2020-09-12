For running the experiments on node2vec one can use the following scripts

Generation of the embeddings: 
./run_gen_stability.sh #(for the main experiment)
./run_gen_walk_length.sh #(for the experiment on the walk length)

And in order to produce the results:
./run_eval_stability.sh
./run_eval_walk_length.sh

The cora dataset will be automatically downloaded and stored within the folder ./data
The blogCatalog dataset has to be copied to ./data/blogCatalog manually

