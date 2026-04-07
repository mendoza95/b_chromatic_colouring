def get_total_m_degree(G):
    number_of_objects = G.number_of_nodes()+G.number_of_edges()
    X = [0 for _ in range(0, number_of_objects+1)]
    Z = [0 for _ in range(0, number_of_objects+1)]
    max_j = 0
    for u in G.nodes(): X[2*G.degree(u)] += 1
    for u,v in G.edges(): X[G.degree(u)+G.degree(v)] += 1
    for j in range(number_of_objects-1, 0, -1): Z[j] = Z[j+1] + X[j]
    for j in range(1, number_of_objects):
        if Z[j-1] >= j:
            max_j = j
    return max_j

def get_index_m_degree(G):
    X = [0 for _ in range(0, len(G.edges())+1)]
    Z = [0 for _ in range(0, len(G.edges())+1)]
    max_j = 0
    for u,v in G.edges(): X[G.degree(u)+G.degree(v)-2] += 1
    for j in range(len(G.edges())-1, 0, -1): Z[j] = Z[j+1] + X[j]
    for j in range(1, len(G.edges())):
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
    n = len(G.nodes())
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