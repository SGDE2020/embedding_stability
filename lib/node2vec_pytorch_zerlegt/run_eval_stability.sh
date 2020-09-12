#!/bin/bash

python3 node2vec_comparison.py --dataset Cora --path embeddings -n 30
python3 node2vec_comparison.py --dataset BlogCatalog --path embeddings -n 30
# uses default values for the rest (like default random seed)
