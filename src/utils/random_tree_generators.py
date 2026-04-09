from disjoint_set import DisjointSet
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import random


def build_random_tree_nodes(n):   
    if n == 1: return {0:0}
    degrees_sum = 0
    nodes_degree = {i:1 for i in range(n)}
    aux = nodes_degree.copy()
    for i in range(n): degrees_sum += nodes_degree[i]
    difference = degrees_sum - 2*(n-1)
    while(difference < 0):
        key = random.choice(list(nodes_degree.keys()))
        nodes_degree[key] += 1
        difference += 1
    #now we decrease the degree of random nodes one by one until we get a proper tree
    for i in range(n): 
        if aux[i]==1: aux.pop(i)
    while(difference > 0):
        key = random.choice(list(aux.keys()))
        aux[key] -= 1
        nodes_degree[key] -= 1
        if aux[key] == 1: aux.pop(key)
        difference -= 1
    assert 2*(n-1) == sum(list(nodes_degree.values()))
    nodes_degree = dict(sorted(nodes_degree.items(), key = lambda item: item[1], reverse=True))
    return nodes_degree

def random_bfs_tree(n):
    nodes_degree = build_random_tree_nodes(n)
    if len(nodes_degree) == 1: 
        T =  nx.Graph()
        T.add_node(list(nodes_degree.keys())[0])
        return T
    number_neighbours = nodes_degree.copy()
    nodes = list(nodes_degree.keys())
    i = 0
    w = nodes[i]
    i += 1
    edge_list = []
    Q = [w]
    while len(Q) != 0:
        u = Q.pop(0)
        for _ in range(number_neighbours[u]):
            v = nodes[i]
            #print(i, (u, v))
            edge_list.append((u, v))
            Q.append(v)
            number_neighbours[v] -= 1
            i += 1
    T = nx.Graph(edge_list)

    return T

def random_recursive_tree(n):
    nodes = [0]
    edge_list = []
    for v in range(1, n):
        u = random.choice(nodes)
        edge_list.append((u, v))
        nodes.append(v)
    return nx.Graph(edge_list)

def random_mst_tree_scipy(n):
    T = nx.complete_graph(n)
    for u, v in T.edges():
        T[u][v]['weight'] = random.random()
    A_t = minimum_spanning_tree(nx.adjacency_matrix(T))
    return nx.from_numpy_array(A_t)

def random_mst_tree(n):
    edges_list = []
    disjoint_set =  DisjointSet(n)
    while len(edges_list) < n-1:
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if disjoint_set.find(i) != disjoint_set.find(j):
            edges_list.append((i,j))
            disjoint_set.union(i, j)
    return nx.Graph(edges_list)

def random_mst_tree_2(n):
    G = nx.complete_graph(n)
    for u, v in G.edges():
        G[u][v]['weight'] = random.random()
    T = nx.minimum_spanning_tree(G)
    return T