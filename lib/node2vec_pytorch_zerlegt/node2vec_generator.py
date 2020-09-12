import torch
import numpy as np
import os
# import os.path 
import sys
import pickle
import torch_geometric
import time
import lz4 # for compressed file support
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm,trange
import random
import argparse

import parse_graph # our library to get along with different types of graph storage types

# time measurement
global_start = time.time()

# initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--path', type=str, default="embeddings", help='Path where the embeddings are stored')
parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs used in training.')
parser.add_argument('-p', '--walk_p', type=float,  default=1.0, help='Parameter p used in the random walks')
parser.add_argument('-q', '--walk_q', type=float,  default=1.0, help='Parameter q used in the random walks')
parser.add_argument('-d', '--dimension', type=int,  default=128, help='Embedding Dimension')
parser.add_argument('--walk_length', type=int, default=None, help='length of the random walks')
parser.add_argument('--walk_seed', type=int,  default=2020, help='Seed for generating the walks')
parser.add_argument('--neg_seed', type=int,  default=2020, help='Seed for negative sampling')
parser.add_argument('--shuffle_seed', type=int,  default=2020, help='Seed for shuffling the epochs\' training data. 0 means no shuffling.')
parser.add_argument('--train_seed', type=int,  default=2020, help='Seed for training the embedding')
parser.add_argument('--dataset', choices=['Cora','BlogCatalog'], default='Cora', help='Which dataset to train on. Currently only Cora or BlogCatalog')
parser.add_argument('--recompute', action='store_true', help='recompute everything, even if it already exists')
parser.add_argument('--temporary_path', type=str, default='tmp_embeddings', help='where to store batches during computation')
parser.add_argument('--dont_pickle', action='store_true', help='set this flag if all computations should be performed in RAM')
# parser.add_argument('--path', type=str, default='embeddings', help='Path where the embeddings are stored')
# negative sampling so far cannot be explicitly seeded

args = parser.parse_args()

# dirty hack to check whether the argument walk_length was set
if args.walk_length == None:
    walk_length = 80
else:
    walk_length = args.walk_length

# global parameters
embedding_dim = args.dimension
epochs = args.epochs




# this function resets all relevant random counters
def reset_random(seed=0):
    random.seed(seed)
    torch.manual_seed(seed) # note that the operation "atomic_add" in torch can still induce slight variations
    np.random.seed(seed)

@torch.no_grad()
def test():
    # model.eval() # sets the model in evaluation mode, equivalent to model.train(False)
    # z = model() # performs a forward pass but with hooks (thats the difference to model.forward())
    # acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     # z[data.test_mask], data.y[data.test_mask], max_iter=150)
    return 0

@torch.no_grad()
def extract_embedding(model):
	model.eval()
	embedding = model(torch.arange(data.num_nodes, device=device))
	return embedding


# casting to torch tensor usually happens in the 'sample' function which is then called in the loader. 
# since the sample function combines positive and negative sampling, we have to avoid it and manually perform the casting 
# for both pos_sample and neg_sample
def my_pos_sample(batch):
    if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
    return model.pos_sample(batch)

def my_neg_sample(batch):
    if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
    return model.neg_sample(batch)

def pickle_batch(batch,walktype,path,epoch):
    filename = os.path.join(path,f'{walktype}_{epoch}.pickle')
    # file = open(filename, mode='wb')
    file = lz4.frame.open(filename, mode='wb')
    pickle.dump(batch,file)
    file.close()
    return filename

def unpickle_batch(filename):
    # file = open(filename,mode='rb')
    file = lz4.frame.open(filename,mode='rb')
    batch = pickle.load(file)
    file.close()
    os.remove(filename)
    return batch

def collect_training_batches(epochs,walk_seed=0, neg_seed=0, pickle=True, pickle_path='.'):
    # a function that ahead of time computes the training batches to analyze effects of randomness

    # positive and negative sampling
    pos_data_loader = torch.utils.data.DataLoader(range(model.adj.sparse_size(0)), collate_fn=my_pos_sample, batch_size=128, shuffle=True, num_workers=0)
    neg_data_loader = torch.utils.data.DataLoader(range(model.adj.sparse_size(0)), collate_fn=my_neg_sample, batch_size=128, shuffle=True, num_workers=0)
    
    # collect positive examples
    reset_random(walk_seed)
    walkstart = time.time()
    pos_batches = []
    for epoch in tqdm(range(epochs),desc='walks      '):
        pos_batch = []
        for pos in pos_data_loader:
            pos_batch.append(pos)
        # either append file handle or the batch directly
        if pickle:
            file = pickle_batch(pos_batch,'walk',pickle_path,epoch)
            pos_batches.append(file)
        else:
            pos_batches.append(pos_batch)
    walkend = time.time()
        
    # collect negative examples
    reset_random(neg_seed)
    neg_time_start = time.time()
    neg_batches = []
    for epoch in tqdm(range(epochs),desc='neg samples'):
        neg_batch = []
        for neg in neg_data_loader:
            neg_batch.append(neg)
        if pickle:
            file = pickle_batch(neg_batch,'neg',pickle_path,epoch)
            neg_batches.append(file)
        else:
            neg_batches.append(neg_batch)
    neg_time_stop = time.time()
        
    print(f'Time for random walks: {walkend - walkstart:.1f}sec, neg sampling: {neg_time_stop - neg_time_start:.1f}sec')
    return pos_batches,neg_batches



def prepare_batch_for_training(pos_batch,neg_batch):
    # combine positive and negative samples pairwise, essentially performing zip manually
    batch = []
    for i in range(len(pos_batch)):
        batch.append([pos_batch[i],neg_batch[i]])
    return batch



# inner training loop. This is where the backpropagation happens
def train_on_single_batch(batch):
    model.train() # equivalent to model.train(True), sets the model in training mode, i.e. enables dropout
    total_loss = 0
    for pos_rw, neg_rw in batch: # the only modification compared to the old train() function
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device)) # node2vec does not need a forward pass so it may just compute the loss directly
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(batch)


# outer training loop, this is where the epochs are handled
def train_on_batches(pos_batches,neg_batches,train_seed=0,pickled=True):    
    starttime = time.time()
    reset_random(train_seed)

    # there are as many batches as there should be epochs
    for current_epoch in range(len(pos_batches)):
        # potentially inflate batch
        if pickled:
            pos_batch = unpickle_batch(pos_batches[current_epoch])
            neg_batch = unpickle_batch(neg_batches[current_epoch])
        else:
            pos_batch = pos_batches[current_epoch]
            neg_batch = neg_batches[current_epoch]
        
        batch = prepare_batch_for_training(pos_batch,neg_batch)

        epochstart = time.time()
        loss = train_on_single_batch(batch)
        acc = test()
        print(f'Epoch: {current_epoch+1:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}, time: {time.time() - epochstart:.1f}sec')
    print(f'Training time: {time.time()-starttime:.1f}sec')
     

# shuffles the epochs' training data
def shuffle_batches(batches,shuffle_seed=0):
    reset_random(shuffle_seed)
    random.shuffle(batches)
    return batches
       

# currently unused plotting function
@torch.no_grad()
def plot_points(embedding,colors=None):
    # plots the embedding using the classes defined in data.y    
    reset_random() # only needed for TSNE
    if colors==None:
        colors = ['#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700']
    print(np.shape(embedding))
    if np.shape(embedding)[1]>2:
        embedding = TSNE(n_components=2).fit_transform(embedding.detach().cpu().numpy())
    y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        plt.scatter(embedding[y == i, 0], embedding[y == i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.show()


# start the actual generation

# compute filename (special naming convention for the walk_length experiment)
csv_filename = f"{args.dataset}_w{args.walk_seed}_n{args.neg_seed}_s{args.shuffle_seed}_t{args.train_seed}.csv.gz"
if args.walk_length != None:
    csv_filename = f"{args.dataset}_w{args.walk_seed}_n{args.neg_seed}_s{args.shuffle_seed}_t{args.train_seed}_len{walk_length}.csv.gz"

# exit if the output file already exists
csv_path = os.path.join(args.path,csv_filename)
if os.path.isfile(csv_path) and not args.recompute:
    print(f"file {csv_filename} aleady exists. Exiting.")
    sys.exit()

# create folder for temporary files
pickle_path = os.path.join(args.temporary_path,f'{args.dataset}_w{args.walk_seed}_n{args.neg_seed}_s{args.shuffle_seed}_t{args.train_seed}_len{walk_length}')
if not os.path.exists(pickle_path) and not args.dont_pickle:
    os.makedirs(pickle_path)


if args.dataset=='Cora': 
    # collect the dataset using pytorch-geometric
    datapath = os.path.join('data', args.dataset)
    dataset = torch_geometric.datasets.Planetoid(datapath, args.dataset)
    data = dataset[0]
elif args.dataset=='BlogCatalog':
    _, blogCatalog_graph = parse_graph.parse_graph_from_dir('data/blogCatalog/')
    data = torch_geometric.utils.from_networkx(blogCatalog_graph)
    # print('chose the blogCatalog dataset')
    # print('shape: '+np.shape(data.edge_index))
else:
    print('dataset not implemented yet')
    data = False

# generate the model, note that other functions directly use the (global) model.
reset_random(0) # old version, the following would be interesting too
# reset_random(args.train_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch_geometric.nn.Node2Vec(data.edge_index, embedding_dim=embedding_dim, walk_length=walk_length,
             context_size=10, walks_per_node=10, num_negative_samples=1, p=args.walk_p, q=args.walk_q,
             sparse=True).to(device)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)


# sample training data for all epochs
pos_batches,neg_batches = collect_training_batches(epochs,walk_seed=args.walk_seed,neg_seed=args.neg_seed,pickle=not args.dont_pickle,pickle_path=pickle_path)

# perform shuffling on the training data, identical shuffling due to reseeding
shuffle_batches(pos_batches,shuffle_seed=args.shuffle_seed) # no shuffling if seed=0
shuffle_batches(neg_batches,shuffle_seed=args.shuffle_seed) # no shuffling if seed=0

# actually compute an embedding
train_on_batches(pos_batches,neg_batches,train_seed=args.train_seed,pickled=not args.dont_pickle)

# store the embedding
embedding = extract_embedding(model).cpu()
header_string = f'trained for {args.epochs} and contains {args.dimension} dimensions. Random seeds are w{args.walk_seed}, n{args.neg_seed}, s{args.shuffle_seed}, and t{args.train_seed}'
np.savetxt(args.path+'/'+csv_filename,embedding,header=header_string)

print(f'successfully saved embedding {csv_filename}. The whole generation took {time.time() - global_start:.1f}sec')
if os.path.isdir(pickle_path):
    os.rmdir(pickle_path)
