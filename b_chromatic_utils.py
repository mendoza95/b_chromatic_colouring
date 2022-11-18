import pickle
import random
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree

def build_random_tree_nodes(n):   
    import random
    if n == 1: return {0:0}
    degrees_sum = 0
    nodes_degree = {i:random.randint(1, n) for i in range(n)}
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
    G = nx.complete_graph(n)
    for u, v in G.edges():
        G[u][v]['weight'] = random.random()
    T = nx.minimum_spanning_tree(G)
    return T

def get_colors_to_use(T, colors, n):
    unique_colors = set(list(colors.values()))
    colors_to_use = {c:get_random_color() for c in unique_colors if c is not None}
    node_colors_list = []
    for u in colors.keys():
        if colors[u] is not None:
            node_colors_list.append(colors_to_use[colors[u]])
        else:
            node_colors_list.append('#FFFFFF')
    return node_colors_list

def get_random_color():
    hexadecimal = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])][0]
    return hexadecimal


def check_b_chromatic_coloring(T, W, colors, m):
    for value in colors.values(): assert value != None
    available_good_colors = {w:set(list(colors.values())) for w in W}
    for w in W:
        available_good_colors[w].discard(colors[w])
        for v in T.adj[w]:
            available_good_colors[w].discard(colors[v])
    for w in W: assert len(available_good_colors[w]) == 0
    for u, v in T.edges(): assert colors[u] != colors[v]

def write_graph_into_txt(T, filename):
    file_graph = open(filename, "a")
    file_graph.write("{} {}\n".format(len(T.nodes()), len(T.edges())))
    for u, v in T.edges():
        file_graph.write("{} {}\n".format(u, v))
    file_graph.close()

def get_edges_lists(file, n_edges):
    edges_list = []
    for i in range(n_edges):
        edge = file.readline().split()
        #print(edge)
        edges_list.append([edge[0], edge[1]])
    return edges_list

def store_graphs_as_list_of_dicts(filename, graph_list):
    #dict_of_trees = {}
    #for key in graph_list.keys():
    #    list_of_dicts = [nx.to_dict_of_lists(G) for G in graph_list[key]]
        #dict_of_trees[key] = list_of_dicts
    list_of_dicts = [nx.to_dict_of_lists(G) for G in graph_list]
    with open(filename, 'wb') as f:
        #pickle.dump(dict_of_trees, f)
        pickle.dump(list_of_dicts, f)

def load_graph_from_list_of_dicts(filename):
    with open(filename, 'rb') as f:
    #    dict_of_trees = pickle.load(f)
        list_of_dicts = pickle.load(f)
    #graphs_dict = {}
    #for key in dict_of_trees.keys():
    #    graphs = [nx.from_dict_of_lists(G) for G in dict_of_trees[key]]
    #    graphs_dict[key] = graphs
    list_of_trees = [nx.from_dict_of_lists(G) for G in list_of_dicts]
    return list_of_trees