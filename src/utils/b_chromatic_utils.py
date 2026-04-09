import networkx as nx
import random

def get_dense_vertices(G, m):
    """Get the dense vertices of a graph G with m-degree m.

    The dense vertices are the vertices with degree at least m-1.

    Args:
        G (networkx.Graph): Graph
        m (int): m-degree of the graph

    Returns:
        list: List of dense vertices
    """
    # check that G is a networkx graph and m is a positive integer
    if (not isinstance(G, nx.Graph)
        or isinstance(G, (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))):
        raise TypeError("G must be a networkx graph")
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    dense_vertices = []
    for u in G.nodes():
        if G.degree(u) >= m-1: dense_vertices.append(u)
    return dense_vertices

def get_biggest_cc(G):
    """
    Get the biggest connected component of a graph G.

    If G is connected, it returns G. Otherwise, it returns the induced subgraph of G corresponding to the biggest connected component.

    Args:
        G (networkx.Graph): Graph
    
    Returns:
        networkx.Graph: Biggest connected component of G
    
    """
    # check that G is a networkx graph    
    if (not isinstance(G, nx.Graph)
        or isinstance(G, (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))):
        raise TypeError("G must be a networkx graph")
    if nx.is_connected(G): return G
    else:
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


def check_b_chromatic_coloring(T, W, colors, m):
    for value in colors.values(): assert value != None
    available_good_colors = {w:set(list(colors.values())) for w in W}
    for w in W:
        available_good_colors[w].discard(colors[w])
        for v in T.adj[w]:
            available_good_colors[w].discard(colors[v])
    for w in W: assert len(available_good_colors[w]) == 0
    assert is_proper(T, colors)
