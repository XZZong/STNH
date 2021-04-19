import argparse
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from texttable import Texttable
from tqdm import tqdm
import random as rd
import logging

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def parameter_parser():
    """
    Parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run SLF.")
    parser.add_argument("--edge-path",
                        nargs="?",
                        default="./data/WikiElec.txt",
                        help="Edge list in txt format.")
    parser.add_argument("--embedding-path",
                        nargs="?",
                        default="./output",
                        help="embedding path.")
    parser.add_argument("--dim",
                        type=int,
                        default=70,
                        help="Dimension of latent factor vector. Default is 70.")
    parser.add_argument("--n",
                        type=int,
                        default=10,
                        help="Number of noise samples. Default is 10.")
    parser.add_argument("--window_size",
                        type=int,
                        default=10,
                        help="Context window size. Default is 10.")
    parser.add_argument("--num_walks",
                        type=int,
                        default=20,
                        help="Walks per node. Default is 20.")
    parser.add_argument("--walk_len",
                        type=int,
                        default=40,
                        help="Length per walk. Default is 40.")
    parser.add_argument("--workers",
                        type=int,
                        default=4,
                        help="Number of threads used for random walking. Default is 4.")
    parser.add_argument("--test-size",
                        type=float,
                        default=0.2,
                        help="Test ratio. Default is 0.2.")
    parser.add_argument("--split-seed",
                        type=int,
                        default=2,
                        help="Random seed for splitting dataset. Default is 2.")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.02,
                        help="Learning rate. Default is 0.02.")
    parser.add_argument("--m",
                        type=float,
                        default=1,
                        help="Damping factor. Default is 1.") # 衰减
    parser.add_argument("--norm",
                        type=float,
                        default=0.01,
                        help="Normalization factor. Default is 0.01.")
    parser.add_argument("--lamb",
                        type=float,
                        default=0.1,
                        help="lambda coefficient. Default is 0.1.")

    return parser.parse_args()


def read_edge_list(args):
    """
    Load edges from a txt file.
    """
    G = nx.DiGraph()
    edges = np.loadtxt(args.edge_path)
    for i in range(edges.shape[0]):
        G.add_edge(int(edges[i][0]), int(edges[i][1]), weight=edges[i][2])
    edges = [[e[0], e[1], e[2]['weight']] for e in G.edges.data()]
    return edges, max(G.nodes) + 1  # index can start from 0.


def parallel_generate_walks(d_graph, walk_len, num_walks, cpu_num):
    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()
    pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        rd.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Start walk
            walk = [source]

            # Perform walk
            while len(walk) < walk_len:

                walk_options = d_graph[walk[-1]]['successors']
                # Skip dead end nodes
                if not walk_options:
                    break

                probabilities = d_graph[walk[-1]]['probabilities']
                walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                walk.append(walk_to)

            # walk with length < 3 does not work in optimization
            if len(walk) > 2:
                walks.append(walk)

    pbar.close()

    return walks

def args_printer(args):
    """
    Print the parameters in tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    t = Texttable()
    l = [[k, args[k]] for k in args.keys()]
    l.insert(0, ["Parameter", "Value"])
    t.add_rows(l)
    print(t.draw())
