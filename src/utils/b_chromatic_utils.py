import networkx as nx

from collections import defaultdict

def check_G_instance(G: nx.Graph):
    """check that G is a networkx graph"""
    if (not isinstance(G, nx.Graph)
        or isinstance(G, (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))):
        raise TypeError("G must be a networkx graph")
    

def get_dense_vertices(G: nx.Graph, m: int) -> list[int]:
    """Get the dense vertices of a graph G with m-degree m.

    The dense vertices are the vertices with degree at least m-1.

    Args:
        G (networkx.Graph): Graph
        m (int): m-degree of the graph

    Returns:
        list[int]: List of dense vertices
    """
    check_G_instance(G)
    if not isinstance(m, int) or m < 0:
        raise ValueError("m must be a positive integer")
    return [u for u in G.nodes() if G.degree(u) >= m-1]

def get_biggest_cc(G: nx.Graph) -> bool | nx.Graph:
    """
    Get the biggest connected component of a graph G.

    If G is connected, it returns G. Otherwise, it returns the induced subgraph of G corresponding to the biggest connected component.

    Args:
        G (networkx.Graph): Graph
    
    Returns:
        bool | networkx.Graph: False if G is the null graph | Biggest connected component of G
    
    """
    check_G_instance(G)
    try:
        if nx.is_connected(G): return G
    except nx.NetworkXPointlessConcept:
        print("G is the null graph, i.e, contains no vertices or edges")
        return False
    else:
        components_dict = {i:cc for i, cc in enumerate(nx.connected_components(G))}
        max_cc = max(components_dict.items(), key=lambda x:len(x[1]))[1]

        G = nx.induced_subgraph(G, max_cc)
    return G

def is_u_b_chromatic(G: nx.Graph, f: dict[int, int], u: int, C: set[int]=None) -> bool:
    """Verify whether vertex u is b-chromatic in G with respect to colouring f
    
    Args:
        G (networkx.Grah): Graph.
        f (dict[int, int]): A partial proper colouring function.
        u (int): A vertex of G.
        C (set[int]): Set of colours used in G minus the colour of u
    
    Returns
        Bool: Whether vertex u is b-chromatic.
    """
    check_G_instance(G)
    if G.number_of_nodes() == 0:
        raise ValueError("G is the null graph")
    if not f:
        raise ValueError("f is empty")
    if u not in G:
        raise KeyError("vertex u is not in G")
    if u not in f:
        raise KeyError("vertex u is not in f")
    
    
    if C is None:
        C = {f[v] for v in G if f[v] is not None}
    k = len(C)
    if len(set({f[v] for v in G.adj[u] if f[v] is not None})) == k-1:
        # If the number of colours in the neighbourhood of u is k-1
        return True
    else:
        return False
    
def are_b_chromatic(G: nx.Graph, f: dict[int, int], V: list[int]) -> bool:
    """Verify whether the vertices in V are b-chromatic vertices with respect to f for some colour c.

    Args:
        G (networkx.Grah): Graph.
        f (dict[int, int]): A partial proper colouring function.
        V (list[int]): set of vertices.
    
    Returns
        Bool: Whether vertices in V are all b-chromatic.
    """
    check_G_instance(G)
    if G.number_of_nodes() == 0:
        raise ValueError("G is the null graph")
    if not f:
        raise ValueError("f is empty")
    if not V:
        raise ValueError("V is empty")
    if any(u not in G for u in V):
        raise KeyError("There is a vertex in V not in G")
    if any(u not in f for u in V):
        raise KeyError("There is a vertex in V not in f")

    C = set([cp for cp in f.values() if cp != None])
    CB = {c:False for c in C if c is not None}
    if all( # For all colours c in C
        any( # There exists a vertex u of colour c such that |c(N(u))| = k-1
            is_u_b_chromatic(G, f, u, C) for u in V if f[u] == c
        ) for c in CB):
        return True
    else:
        return False

def is_proper(G: nx.Graph, f: dict[int, int]) -> bool:
    """Verify whether a colouring c is proper and not partial (every vertex is coloured).

    A proper colouring is a an assignment of colours (labels) to the vertices of V such that any pair of adjacent vertices are assigned different colours.

    Args:
        G (networkx.Graph): Graph
        f (dict[int, int]): Colouring function
        
    Return:
        bool: Whether c is proper and not partial.    
    """
    check_G_instance(G)
    if G.number_of_nodes() == 0:
        raise ValueError("G is the null graph")
    if not f:
        raise ValueError("f is empty")
    
    if all(
        f[u] is not None and f[v] is not None and f[u] != f[v] for u, v in G.edges()
    ):
        return True
    else:
        return False

def get_b_chromatic_vertices(G: nx.Graph, f: dict[int, int]) -> dict[int, list[int]]:
    """"Get the b-chromatic vertices of a graph G with respect to a partial proper colouring c
    
    Args:
        G (networkx.Graph): Graph.
        f (dict[int,int]): Colouring function.
    
    Returns:
        dict[int, list[int]]: Dictionary of colour classes, i.e., each colour key is associated 
        to a list of b-chromatic vertex of that colour class.
    """
    check_G_instance(G)
    if G.number_of_nodes() == 0:
        raise ValueError("G is the null graph")
    if not f:
        raise ValueError("f is empty")
    
    C = {c:[] for c in f.values() if c is not None}
    b_colour_classes = {c:[] for c in C}
    for u in G.nodes():
        if is_u_b_chromatic(G, f, u, C):
            b_colour_classes[f[u]].append(u)
    return b_colour_classes


def compute_remaining_colors(G, W, f, C=None):
    """Get the remaining colours vertices in W need to pick up to become b-chromatic.
    Args:
        G (networkx.Graph): Graph
        W (list[int]): Vertices for which the remaining colours need to be computed
        f (dict[int, int]): Colouring funtion
        C (set[int]): Set of colours used by f

    Returns:
        dict[int, set[int]]: Remaining colours that vertex w in W need to pick up to become b-chromatic
    """
    check_G_instance(G)
    if G.number_of_nodes() == 0:
        raise ValueError("G is the null graph")
    if not f:
        raise ValueError("f is empty")
    if C is None:
        C = set([c for c in f.values() if c is not None])

    good_set_remaining_colors = {w:C.copy() for w in W}
    for w in W:
        good_set_remaining_colors[w].discard(f[w])
        for v in G.adj[w]:
            if f[v] is not None:
                good_set_remaining_colors[w].discard(f[v])
    return good_set_remaining_colors

def pick_available_color(G, u, f, C=None) -> int | None:
    """Picks an available color for a given vertex u from a set of available colors.

    This function determines a color for vertex `u` such that it does not conflict
    with the colors of its already colored neighbors, adhering to proper coloring rules.

    Args:
        T (networkx.Graph): The graph.
        u (int): The vertex for which to pick an available color.
        colours (dict[int, int | None]): A dictionary mapping vertices to their assigned
                                         colors (or None if uncolored).
        available_colours (set[int]): A set of all colors that are potentially available for coloring.

    Returns:
        int: An available color for vertex `u` that satisfies proper coloring constraints; None if no colour is available."""
    
    check_G_instance(G)
    if G.number_of_nodes() == 0:
        raise ValueError("G is the null graph")
    if not f:
        raise ValueError("f is empty")
    if C is None:
        C = set([c for c in f.values() if c is not None])

    
    colourhood_u = {f[v] for v in G.adj[u] if f[v]}
    for c in C:
        if c not in colourhood_u:
            return c
    return None
    #available_colours_u = {c:True for c in C}
    #for v in G.adj[u]:
    #    if f[v] is not None:
    #        available_colours_u[f[v]] = False
    #for c in available_colours_u.keys():
    #    if available_colours_u[c]: return c
    

def check_b_chromatic_coloring(T, W, colors, m):
    # This function is the same as are_b_chromatic function so it must be discarded.
    for value in colors.values(): assert value != None
    available_good_colors = {w:set(list(colors.values())) for w in W}
    for w in W:
        available_good_colors[w].discard(colors[w])
        for v in T.adj[w]:
            available_good_colors[w].discard(colors[v])
    for w in W: assert len(available_good_colors[w]) == 0
    assert is_proper(T, colors)
