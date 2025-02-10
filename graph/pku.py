import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 20
self_link = [(i, i) for i in range(num_node)]
inward = [(0, 17), (1, 0), (2, 0), (3, 1), (4, 2), (5, 17),(6, 17), (7, 5), 
                    (8, 6), (9, 7), (10, 8), (11, 19),(12, 19), (13, 11), (14, 12), 
                    (15, 13), (16, 14), (18, 17),(19, 18)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_binary = tools.edge2mat(neighbor, num_node)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)
        self.A_outward_binary = tools.get_adjacency_matrix(self.outward, self.num_node)


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'spatialnext':
            A = tools.get_spatial_graphnext(num_node, self_link, inward, outward)
        elif labeling_mode == 'spatial_intensive':
            A = tools.get_ins_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
    
class Graph_HDGCN:
    def __init__(self, CoM=17, labeling_mode='spatial'):
        self.num_node = 20
        self.CoM = CoM
        self.A = self.get_adjacency_matrix(labeling_mode)
        

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_hierarchical_graph(self.num_node, tools.get_edgeset(dataset='pku', CoM=self.CoM)) # L, 3, 20, 20
        else:
            raise ValueError()
        return A, self.CoM

class Graph_STAM:
    def __init__(self,
                 max_hop=3,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node = 20
        neighbor_link = inward
        self.edge = self_link + neighbor_link

        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency()
    def get_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        A = np.zeros((1, self.num_node, self.num_node))
        A[0] = normalize_adjacency
        self.A = A

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD