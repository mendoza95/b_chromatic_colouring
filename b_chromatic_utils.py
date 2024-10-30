import pickle
import random
import networkx as nx
from disjoint_set import DisjointSet
from scipy.sparse.csgraph import minimum_spanning_tree


def get_m_degree(G):
    X = [0 for _ in range(0, len(G.nodes())+1)]
    Z = [0 for _ in range(0, len(G.nodes())+1)]
    max_j = 0
    for u in G.nodes(): X[G.degree(u)] += 1
    for j in range(len(G.nodes())-1, 0, -1): Z[j] = Z[j+1] + X[j]
    for j in range(1, len(G.nodes)):
        if Z[j-1] >= j:
            max_j = j
    return max_j

def get_vertices_sorted_by_degree(G):
    vertex_degree_list = {u:G.degree(u) for u in G.nodes()}
    sorted_vertex_set = dict(sorted(vertex_degree_list.items(), key=lambda item:item[1], reverse=True))
    return sorted_vertex_set

def get_m_degree_2(G):
    sorted_vertex_set = get_vertices_sorted_by_degree(G)
    m = 0
    for i, di in enumerate(sorted_vertex_set.values()):
        if (di >= i): m = i + 1
    return m

def get_dense_vertices(G, m):
    dense_vertices = []
    for u in G.nodes():
        if G.degree(u) >= m-1: dense_vertices.append(u)
    return dense_vertices

def get_biggest_cc(G):
    if not nx.is_connected(G):
        CC = list(nx.connected_components(G))
        i_max = 0
        for i, cc in enumerate(CC):
            if len(cc) >= len(CC[i_max]):
                i_max = i
        CCmax = CC[i_max]
        G = nx.induced_subgraph(G, CCmax)
    return G

def are_b_chromatic(G, f, V):
    """Check whether the vertices in V are b-chromatic for every colour in c
        G: Graph 
        f: colouring function, 
        V: set of vertices
    """
    C = set([cp for cp in f.values() if cp != None])
    #flag for checking if there is a b-chromatic vertex for colour c
    CB = {c:False for c in C if c is not None}
    k = len(CB)
    for c in CB:
        for u in V:
            if f[u] == c:
                Cp = set()
                for v in G.adj[u]:
                    Cp.add(f[v])
                if len(Cp) == k-1: CB[c] = True
    for u in V:
        if not CB[f[u]]: return False
    return True

def is_proper(G, c):
    """Check whether the colouring c is proper
    G: Graph, c: colour assignment function"""
    for u,v in G.edges():
        if((c[u] is not None and c[v] is not None) and (c[u] == c[v])):
            return False
    return True

def get_b_chromatic_vertices(G, c):
    """"Get the b-chromatic vertices from a graph G based on the colouring c
    G: Graph, c: colour assignment function"""
    C = {d for d in c.values() if d is not None}
    B = []
    for u in G.nodes():
        cNu = {c[v] for v in G.adj[u] if c[v] is not None}
        if len(cNu) == len(C)-1: B.append(u)
    return B

def build_random_tree_nodes(n):   
    import random
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

def get_colors_to_use(colors):
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
    assert is_proper(T, colors)

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