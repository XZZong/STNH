import numpy as np
from collections import defaultdict
class Graphlet(object):
    def __init__(self, G, n):
        self.G = G
        self.n = n # 节点个数
        
    def init(self, edges):
        # set up adjacency matrix if it's smaller than 100MB
        if self.n * self.n < 100 * 1024 * 1024 * 8:
            self.adj_chunk = 8 * 4 # sizeof(int)
            self.adj_matrix = np.zeros((int(self.n * self.n / self.adj_chunk) + 1), dtype=int)
            for edge in edges:
                x, y, _ = edge
                self.adj_matrix[(x * self.n + y) // self.adj_chunk] |= (1<<((x * self.n + y) % self.adj_chunk))
            self.adjacent = lambda x, y : x != y and not self.adj_matrix[(x * self.n + y)//self.adj_chunk]&(1<<(x * self.n + y)%self.adj_chunk)\
                                        and not self.adj_matrix[(y * self.n + x)//self.adj_chunk]&(1<<(y * self.n + x)%self.adj_chunk)
        else:
            self.adjacent = lambda x, y : x != y and x not in list(self.G.successors(y)) and y not in list(self.G.successors(x))
        
    # def count(self, edges):
    def preparing(self, edges):
        print("preparing data")
        pos_out_edge_set = defaultdict(set)
        neg_out_edge_set = defaultdict(set)
        pos_in_edge_set = defaultdict(set)
        neg_in_edge_set = defaultdict(set)
        for edge in edges:
            x, y, z = edge
            if z > 0:
                pos_out_edge_set[x].add(y)
                pos_in_edge_set[y].add(x)
            elif z < 0:
                neg_out_edge_set[x].add(y)
                neg_in_edge_set[y].add(x)
        return pos_out_edge_set, pos_in_edge_set, neg_out_edge_set, neg_in_edge_set

    def count(self, edges):
        pos_out_edge_set, pos_in_edge_set, neg_out_edge_set, neg_in_edge_set = self.preparing(edges)
        print("stage1 - common neighbor calculation")
        neighbors = np.zeros((len(edges), 24), dtype=int)
        # 重复计算了，应该是针对存在边的节点对，但如果边集中uv有 vu基本没有，那这样处理也可以
        for i, edge in enumerate(edges): # TODO
            x, y, _ = edge
            case = 0
            for x_set in (pos_out_edge_set[x], pos_in_edge_set[x], neg_out_edge_set[x], neg_in_edge_set[x]):
                for y_set in (pos_out_edge_set[y], pos_in_edge_set[y], neg_out_edge_set[y], neg_in_edge_set[y]):
                    neighbors[i][case] = len(x_set.intersection(y_set))
                    case += 1
            # 如果有的点同时存在在多个中，下面的减法运算的结果可能为负
            neighbors[i][16] = max(len(pos_out_edge_set[y]) - neighbors[i][0] - neighbors[i][4] - neighbors[i][8] - neighbors[i][12], 0)
            neighbors[i][17] = max(len(pos_in_edge_set[y]) - neighbors[i][1] - neighbors[i][5] - neighbors[i][9] - neighbors[i][13], 0)
            neighbors[i][18] = max(len(neg_out_edge_set[y]) - neighbors[i][2] - neighbors[i][6] - neighbors[i][10] - neighbors[i][14], 0)
            neighbors[i][19] = max(len(neg_in_edge_set[y]) - neighbors[i][3] - neighbors[i][7] - neighbors[i][11] - neighbors[i][15], 0)
            neighbors[i][20] = max(len(pos_out_edge_set[x]) - neighbors[i][0] - neighbors[i][1] - neighbors[i][2] - neighbors[i][3], 0)
            neighbors[i][21] = max(len(pos_in_edge_set[x]) - neighbors[i][4] - neighbors[i][5] - neighbors[i][6] - neighbors[i][7], 0)
            neighbors[i][22] = max(len(neg_out_edge_set[x]) - neighbors[i][8] - neighbors[i][9] - neighbors[i][10] - neighbors[i][11], 0)
            neighbors[i][23] = max(len(neg_in_edge_set[x]) - neighbors[i][12] - neighbors[i][13] - neighbors[i][14] - neighbors[i][15], 0)

        print("stage2 - full graphlet")
        orbit = np.zeros((self.n, 58), dtype=int)
        for i, edge in enumerate(edges): # TODO
            x, y, z = edge
            if z > 0:
                orbit[x][35] += neighbors[i][0] + neighbors[i][1] # x、y的pos_out的交集, x的pos_out 与 y的pos_in的交集
                orbit[x][26] += neighbors[i][4] # x的pos_in 与 y的pos_out的交集
                orbit[x][38] += neighbors[i][8] # x的neg_out 与 y的pos_out
                orbit[x][28] += neighbors[i][12] # x的neg_in 与 y的pos_out
                orbit[x][36] += neighbors[i][5] # x、y的pos_in的交集
                orbit[x][44] += neighbors[i][9]
                orbit[x][45] += neighbors[i][13]
                orbit[x][41] += neighbors[i][2] + neighbors[i][3]
                orbit[x][29] += neighbors[i][6]
                orbit[x][47] += neighbors[i][10]
                orbit[x][31] += neighbors[i][14]
                orbit[x][39] += neighbors[i][7]
                orbit[x][53] += neighbors[i][11]
                orbit[x][51] += neighbors[i][15]
                orbit[x][0]  += neighbors[i][16] # pos_out_y_left
                orbit[x][12] += neighbors[i][17] # pos_in_y_left
                orbit[x][6]  += neighbors[i][18] # neg_out_y_left
                orbit[x][16] += neighbors[i][19] # neg_in_y_left
                orbit[x][20] += neighbors[i][20]
                orbit[x][1]  += neighbors[i][21]
                orbit[x][22] += neighbors[i][22]
                orbit[x][4]  += neighbors[i][23]
                # -------------------------------------- #
                orbit[y][36] += neighbors[i][0] 
                orbit[y][34] += neighbors[i][1] + neighbors[i][5]
                orbit[y][42] += neighbors[i][2]
                orbit[y][40] += neighbors[i][3]
                orbit[y][26] += neighbors[i][4]
                orbit[y][27] += neighbors[i][6]
                orbit[y][37] += neighbors[i][7]
                orbit[y][39] += neighbors[i][8]
                orbit[y][43] += neighbors[i][9] + neighbors[i][13]
                orbit[y][48] += neighbors[i][10]
                orbit[y][52] += neighbors[i][11]
                orbit[y][29] += neighbors[i][12]
                orbit[y][32] += neighbors[i][14]
                orbit[y][49] += neighbors[i][15]
                orbit[y][1]  += neighbors[i][16]
                orbit[y][13] += neighbors[i][17]
                orbit[y][7]  += neighbors[i][18]
                orbit[y][15] += neighbors[i][19]
                orbit[y][19] += neighbors[i][20] # pos_out_x_left
                orbit[y][2]  += neighbors[i][21] # pos_in_x_left
                orbit[y][23] += neighbors[i][22] # neg_out_x_left
                orbit[y][5]  += neighbors[i][23] # neg_in_x_left
            else:
                orbit[x][44] += neighbors[i][0]
                orbit[x][27] += neighbors[i][4]
                orbit[x][50] += neighbors[i][8] + neighbors[i][9]
                orbit[x][30] += neighbors[i][12]
                orbit[x][38] += neighbors[i][1]
                orbit[x][42] += neighbors[i][5] # x、y的pos_in的交集
                orbit[x][54] += neighbors[i][13]
                orbit[x][53] += neighbors[i][2]
                orbit[x][32] += neighbors[i][6]
                orbit[x][56] += neighbors[i][10]
                orbit[x][33] += neighbors[i][14]
                orbit[x][47] += neighbors[i][3]
                orbit[x][48] += neighbors[i][7]
                orbit[x][56] += neighbors[i][11]
                orbit[x][57] += neighbors[i][15]
                orbit[x][3]  += neighbors[i][16] # pos_out_y_left
                orbit[x][14] += neighbors[i][17] # pos_in_y_left
                orbit[x][9]  += neighbors[i][18] # neg_out_y_left
                orbit[x][17] += neighbors[i][19] # neg_in_y_left
                orbit[x][22] += neighbors[i][20]
                orbit[x][7]  += neighbors[i][21]
                orbit[x][25] += neighbors[i][22]
                orbit[x][10] += neighbors[i][23]
                # -------------------------------------- #
                orbit[y][45] += neighbors[i][0] 
                orbit[y][37] += neighbors[i][1]
                orbit[y][54] += neighbors[i][2]
                orbit[y][46] += neighbors[i][3] + neighbors[i][7]
                orbit[y][28] += neighbors[i][4]
                orbit[y][40] += neighbors[i][5]
                orbit[y][30] += neighbors[i][6]
                orbit[y][51] += neighbors[i][8] 
                orbit[y][49] += neighbors[i][9]
                orbit[y][57] += neighbors[i][10]
                orbit[y][55] += neighbors[i][11] + neighbors[i][15]
                orbit[y][31] += neighbors[i][12]
                orbit[y][52] += neighbors[i][13]
                orbit[y][33] += neighbors[i][14]
                orbit[y][4]  += neighbors[i][16]
                orbit[y][15] += neighbors[i][17]
                orbit[y][10] += neighbors[i][18]
                orbit[y][18] += neighbors[i][19]
                orbit[y][21] += neighbors[i][20] # pos_out_x_left
                orbit[y][8]  += neighbors[i][21] # pos_in_x_left
                orbit[y][24] += neighbors[i][22] # neg_out_x_left
                orbit[y][11] += neighbors[i][23] # neg_in_x_left
            # 不少情况下，点属于两条边，分别计算了一遍，出现重复
        once = [0, 2, 3, 5, 6, 8, 9, 11, 14, 16, 21, 23]
        orbit[:, once] *= 2
        orbit //= 2
        return orbit

    # 通过不同的计算方式，验证前面的计算结果的正确性
    def count2(self, edges):
        self.init(edges)
        pos_out_edge_set, pos_in_edge_set, neg_out_edge_set, neg_in_edge_set = self.preparing(edges)
        test_nodes = np.random.randint(0, self.n, 8)
        for node in test_nodes:
            orbit = np.zeros((6), dtype=int)
            for v in pos_out_edge_set[node]:
                for t in pos_in_edge_set[node]: # node的in、out中不是邻居，这种计算不存在重复
                    if self.adjacent(v, t): # 1
                        orbit[0] += 1
                for t in neg_in_edge_set[node]:
                    if self.adjacent(v, t): # 4
                        orbit[1] += 1
                for t in neg_out_edge_set[node]:
                    if self.adjacent(v, t): # 22
                        orbit[5] += 1
            for v in neg_out_edge_set[node]:
                for t in pos_in_edge_set[node]:
                    if self.adjacent(v, t): # 7
                        orbit[2] += 1
                for t in neg_in_edge_set[node]:
                    if self.adjacent(v, t): # 10
                        orbit[3] += 1
            for v in pos_in_edge_set[node]:
                for t in neg_in_edge_set[node]:
                    if self.adjacent(v, t): # 15
                        orbit[4] += 1
            print(node, orbit)