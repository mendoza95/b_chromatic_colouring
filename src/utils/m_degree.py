import networkx as nx

def get_total_m_degree(G):
    """Computes the total m-degree of a graph G.
    
    This function takes a graph G as input and computes its total m-degree, which is the maximum integer j such that there exists at least j elements in G with total degree at least j-1.
    An element of G can be either a node or an edge.
    The total degree of an element x is twice its degree if x is a node, and it is the sum of the degrees of its endpoints if x is an edge.

    Args:
        G (networkx.Graph): A networkx graph.

    Returns:
        max_j (int): The total m-degree of G.
    """
    if (
        not isinstance(G, nx.Graph) or 
        isinstance(G, (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
    ):
        raise TypeError("Input must be a networkx Graph.")  
    number_of_objects = G.number_of_nodes()+G.number_of_edges()
    if number_of_objects == 0: return 0
    X = [0 for _ in range(0, number_of_objects+1)] # Stores the number of elements with total degree i
    Z = [0 for _ in range(0, number_of_objects+1)] # Stores the number of elements with total degree at least i
    max_j = 0
    for u in G.nodes(): X[2*G.degree(u)] += 1
    for u,v in G.edges(): X[G.degree(u)+G.degree(v)] += 1
    Z[number_of_objects-1] = X[number_of_objects-1]
    for j in range(number_of_objects-2, -1, -1): Z[j] = Z[j+1] + X[j]
    for j in range(1, number_of_objects+1):
        if Z[j-1] >= j:
            max_j = j
    return max_j

def get_index_m_degree(G):
    """Computes the index m-degree of a graph G.
    
    This function takes a graph G as input and computes its index m-degree, which is the maximum integer j such that there exists at least j edges in G with degree at least j-1.
    The degree of an edge (u,v) is defined as the sum of the degrees of its endpoints minus 2.
    
    Args:
        G (networkx.Graph): A networkx graph.

    Returns:
        max_j (int): The index m-degree of G.
    """
    if (
        not isinstance(G, nx.Graph) or 
        isinstance(G, (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
    ):
        raise TypeError("Input must be a networkx Graph.")
    n = G.number_of_edges()
    if n == 0: return 0
    X = [0 for _ in range(0, n)] # Stores the number of edges with degree i
    Z = [0 for _ in range(0, n)] # Stores the number of edges with degree at least i
    max_j = 0
    for u,v in G.edges(): X[G.degree(u)+G.degree(v)-2] += 1
    Z[n-1] = X[n-1]
    for j in range(n-2, -1, -1): Z[j] = Z[j+1] + X[j]
    for j in range(1, n+1):
        if Z[j-1] >= j:
            max_j = j
    return max_j

def get_m_degree(G):
    """Computes the m-degree of a graph G.

    This function takes a graph G as input and computes its m-degree, which is the maximum integer j such that there exists at least j vertices in G with degree at least j-1.

    Args:
        G (networkx.Graph): A networkx graph.

    Returns:
        max_j (int): The m-degree of G.
    """
    if (
        not isinstance(G, nx.Graph) or 
        isinstance(G, (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
    ):
        raise TypeError("Input must be a networkx Graph.")
    n = G.number_of_nodes()
    if n == 0: return 0
    X = [0 for _ in range(0, n)] # Stores the number of vertices with degree i
    Z = [0 for _ in range(0, n)] # Stores the number of vertices with degree at least i
    max_j = 0
    for u in G.nodes(): X[G.degree(u)] += 1
    Z[n-1] = X[n-1]
    for j in range(n-2, -1, -1): Z[j] = Z[j+1] + X[j]
    for j in range(1, n+1):
        if Z[j-1] >= j:
            max_j = j
    return max_j