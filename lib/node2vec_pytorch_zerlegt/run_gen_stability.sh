#! /bin/bash
export embeddings='30'

export epochs='10'
#export inmemory=''
export inmemory='--dont_pickle'
export overwrite=''
export workers='1'

export node2vecdataset='BlogCatalog'
export global="--dataset ${node2vecdataset}   -e ${epochs} ${inmemory} ${overwrite} "
echo $global
seq $embeddings | xargs -P $workers -I @ sh -c '{ echo base @ ; python node2vec_generator.py --walk_seed=@ --neg_seed=$(( 100+@ )) --shuffle_seed=$(( 200+@ )) --train_seed=$(( 300+@ )) $global; }'
seq $embeddings | xargs -P $workers -I @ sh -c '{ echo walk @ ; python node2vec_generator.py --walk_seed=@ $global; }'
seq $embeddings | xargs -P $workers -I @ sh -c '{ echo neg @ ; python node2vec_generator.py --neg_seed=@ $global; }'
seq $embeddings | xargs -P $workers -I @ sh -c '{ echo shuffle @ ; python node2vec_generator.py --shuffle_seed=@ $global; }'

export node2vecdataset='Cora'
export global="--dataset ${node2vecdataset}   -e ${epochs} ${inmemory} ${overwrite} "

seq $embeddings | xargs -P $workers -I @ sh -c '{ echo base @ ; python node2vec_generator.py --walk_seed=@ --neg_seed=$(( 100+@ )) --shuffle_seed=$(( 200+@ )) --train_seed=$(( 300+@ )) $global; }'
seq $embeddings | xargs -P $workers -I @ sh -c '{ echo walk @ ; python node2vec_generator.py --walk_seed=@ $global; }'
seq $embeddings | xargs -P $workers -I @ sh -c '{ echo neg @ ; python node2vec_generator.py --neg_seed=@ $global; }'
seq $embeddings | xargs -P $workers -I @ sh -c '{ echo shuffle @ ; python node2vec_generator.py --shuffle_seed=@ $global; }'



#for i in `seq $embeddings`; do echo walk $i ; python node2vec_generator.py --walk_seed=$i --dataset $node2vecdataset -e $epochs; done
#export global="--dataset ${node2vecdataset}   -e ${epochs} ${inmemory} ${overwrite} "
#for i in `seq $embeddings`; do echo neg $i ; python node2vec_generator.py --neg_seed=$i --dataset $node2vecdataset -e $epochs; done
#for i in `seq $embeddings`; do echo shuffle $i ; python node2vec_generator.py --shuffle_seed=$i --dataset $node2vecdataset -e $epochs; done
