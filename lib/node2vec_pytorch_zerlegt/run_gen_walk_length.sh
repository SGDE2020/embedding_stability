#! /bin/bash
export embeddings='10'
export min_walk_length='10'
export max_walk_length='150'
export walk_increment='5'
#export recompute='--recompute'
export recompute=''
export inmemory='--dont_pickle'

export node2vecdataset='Cora'
#export epochs='10'
export total_examples='1500'
export workers='1'


#for i in `seq $embeddings`; do echo length $length walk $i ; python node2vec_generator.py --walk_seed=$i --dataset $node2vecdataset -e $(( $total_examples / $length )) -s embeddings_walk_length_long $recompute --walk_length $length; done

for length in `seq $min_walk_length $walk_increment $max_walk_length`; do
	export global="--dataset $node2vecdataset -s embeddings_walk_length $recompute $inmemory -e $(( $total_examples / $length )) --walk_length $length "
	seq $embeddings | xargs -P $workers -I @ sh -c '{ echo length $length walk @ ; python node2vec_generator.py --walk_seed=@ $global; }' 
done
