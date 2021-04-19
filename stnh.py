"""
    Social Trust Network Embedding with Hashing and graphlet (STNH)

    Default dimension is 256 (128 for out embedding and 128 for in embedding)
    For link prediction, we use 80% edges for training.
    The running speed is slow! We are trying to use TensorFlow to rewrite it.
"""

import networkx as nx
import random
import numpy as np
from collections import defaultdict
from utils import read_edge_list, parallel_generate_walks, sigmoid
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import random as rd
from tqdm import tqdm
from graphlet import Graphlet

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

class WeightedAggregator():
    def aggregate(self, vecs, weights):
        output = weights @ vecs
        return output

    def compute_gradients(self, vecs, weights, embedding):
        vecs_grad = np.array([embedding * p for p in weights])
        p_grad = vecs @ embedding
        return vecs_grad, p_grad

class STNH(object):
    def __init__(self, args):
        self.args = args
        self.logs = {'epoch': [], 'sign_prediction_auc': [], 'sign_prediction_macro_f1': []}
        self.edges, self.num_nodes, self.train_edges, self.test_edges, self.G, self.d_graph = self._setup()
        self.dataset_name = self.getDatasetName()
        self.aggregator = WeightedAggregator()

        self.num_buckets = int(self.num_nodes / 10)
        self.node_count = self.num_nodes # 节点总数
        self.num_hash_functions = 2
        self.tp_dim = 58

        # Initialize hash table. Note that this could easily be implemented by a modulo operation
        self.hash_tables = np.random.randint(0, 2 ** 30, size=(self.node_count, self.num_hash_functions)) % self.num_buckets
        # Initialize word importance parameters
        self.p_init_std = 0.0005
        self.p_hash_out = np.random.normal(0, self.p_init_std, (self.node_count, self.num_hash_functions))
        self.p_hash_in = np.random.normal(0, self.p_init_std, (self.node_count, self.num_hash_functions))

        #Initialize the embedding matrix
        # latent factor vectors.
        self.lf_out = np.random.rand(self.num_buckets, self.args.dim)
        self.lf_in = np.random.rand(self.num_buckets, self.args.dim)
        # Outward node role vectors.
        self.out_tp = np.zeros((self.num_nodes, self.tp_dim))
        # Inward node role vectors.
        self.in_tp = np.zeros((self.num_nodes, self.tp_dim))

        self.graphlet = Graphlet(self.G, self.num_nodes)
        self.orbit = self.graphlet.count(self.train_edges)
        self.orbit = np.log10(self.orbit + 1) * self.args.lamb
        print("complete graphlet counting")
        self.walks = self._generate_walks()

    def getDatasetName(self):
        path = self.args.edge_path
        name = path.split('/')[-1]
        index = name.index('.')
        return name[:index]

    def _setup(self):
        edges, num_nodes = read_edge_list(self.args)
        train_edges, test_edges = train_test_split(edges,
                                                   test_size=self.args.test_size,
                                                   random_state=self.args.split_seed)
        d_graph = defaultdict(dict)
        G = nx.DiGraph()
        for edge in train_edges:
            if edge[2] > 0:
                G.add_edge(edge[0], edge[1], weight=edge[2], polarity=1)
            elif edge[2] < 0:
                G.add_edge(edge[0], edge[1], weight=abs(edge[2]), polarity=-1)
        for node in G.nodes():
            unnormalized_weights = []
            succs = list(G.successors(node)) # 后继节点

            if not succs:
                d_graph[node]['probabilities'] = []
                d_graph[node]['successors'] = []
            else:
                for succ in succs:
                    weight = G[node][succ]['weight']
                    unnormalized_weights.append(weight)
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[node]['probabilities'] = unnormalized_weights / unnormalized_weights.sum()
                d_graph[node]['successors'] = succs

        return edges, num_nodes, train_edges, test_edges, G, d_graph

    def _generate_walks(self):
        flatten = lambda l: [item for sublist in l for item in sublist]
        num_walks_lists = np.array_split(range(self.args.num_walks), self.args.workers)

        walk_results = Parallel(n_jobs=self.args.workers)(
            delayed(parallel_generate_walks)(self.d_graph,
                                             self.args.walk_len,
                                             len(num_walks),
                                             idx, ) for
            idx, num_walks in enumerate(num_walks_lists))

        return flatten(walk_results)

    # 转为单位向量
    def normalization(self, data):
        if (data==0).all():
            return data
        return data / np.sqrt(np.sum(data ** 2))

    def hash(self, node, center):
        # node = node % self.node_count
        if center:
            w = self.lf_in[self.hash_tables[node], :]
            p = self.p_hash_in[node, :]
        else:
            w = self.lf_out[self.hash_tables[node], :]
            p = self.p_hash_out[node, :]
        return w, p
    
    def hashEmbedding(self, node, center):
        w, p = self.hash(node, center)
        return self.aggregator.aggregate(w, p)

    def updateLF(self, node, HE, center):
        # node = node % self.node_count
        w, p = self.hash(node, center)
        lf_grad, p_grad = self.aggregator.compute_gradients(w, p, HE)
        w += self.args.learning_rate * (lf_grad - self.args.norm * w)
        p += self.args.learning_rate * (p_grad - self.args.norm * p)
        if center:
            self.lf_in[self.hash_tables[node]] = w
            self.p_hash_in[node] = self.normalization(p)
        else:
            self.lf_out[self.hash_tables[node]] = w
            self.p_hash_out[node] = self.normalization(p)

    def fit(self):
        path = '%s/%s-%g-' % (self.args.embedding_path, self.dataset_name, self.args.test_size)
        pbar = tqdm(total=len(self.walks), desc='Optimizing', ncols=100)
        nodes = list(self.d_graph.keys())
        rd.shuffle(self.walks)
        fac = 10
        
        for walk in self.walks:
            pbar.update(1)
            walk_len = len(walk)
            for start in range(walk_len - 1):
                u = walk[start]
                sign = 1
                context = walk[start + 1: min(start + self.args.window_size + 1, self.args.walk_len)]
                pre_v = u
                path_len = 0
                for v in context:
                    if v == u:
                        break
                    sign *= self.G[pre_v][v]['polarity']
                    trust = ((self.args.window_size - path_len) / self.args.window_size) ** self.args.m # m 衰减
                    path_len += 1

                    HE_u = self.hashEmbedding(u, False)
                    HE_v = self.hashEmbedding(v, True)
                    X = HE_u @ HE_v / fac + self.out_tp[u] @ self.orbit[v] + self.in_tp[v] @ self.orbit[u]
                    p_uv = sigmoid(X)
                    Lu = (sign * trust * (1 - p_uv)) * HE_v
                    Tp = (sign * trust * (1 - p_uv)) * self.orbit[v]
                    self.updateLF(v, sign * trust * (1 - p_uv) / fac * HE_u, True)
                    self.in_tp[v] += self.args.learning_rate * ((sign * trust * (1 - p_uv)) * self.orbit[u] - self.args.norm * self.in_tp[v])
                    pre_v = v
                    
                    # negative sampling
                    for i in range(self.args.n):
                        noise = random.choice(nodes)
                        HE_noise = self.hashEmbedding(noise, True)
                        X = HE_u @ HE_noise / fac + self.out_tp[u] @ self.orbit[noise] + self.in_tp[noise] @ self.orbit[u]
                        p_noise = sigmoid(-sign * X)
                        Lu += (-sign * trust * (1 - p_noise)) * HE_noise
                        Tp += (-sign * trust * (1 - p_noise)) * self.orbit[noise]
                        self.updateLF(noise, -sign * trust * (1 - p_noise) / fac * HE_u, True)
                        self.in_tp[noise] += self.args.learning_rate * ((-sign * trust * (1 - p_noise)) * self.orbit[u] - self.args.norm * self.in_tp[noise])
                    self.updateLF(u, Lu / fac, False)
                    self.out_tp[u] += self.args.learning_rate * (Tp - self.args.norm * self.out_tp[u])
        pbar.close()

        W_in = np.zeros((self.num_nodes, self.args.dim + self.tp_dim))
        W_out = np.zeros((self.num_nodes, self.args.dim + self.tp_dim))
        for i in range(self.num_nodes):
            W_in[i, : self.args.dim] = self.hashEmbedding(i, True) 
            W_in[i, self.args.dim :] = self.in_tp[i]
            W_out[i, : self.args.dim] = self.hashEmbedding(i, False)
            W_out[i, self.args.dim : ] = self.out_tp[i]

        np.save(path + "in" , W_in)
        np.save(path + "out", W_out)
        print("saving embedding")
    