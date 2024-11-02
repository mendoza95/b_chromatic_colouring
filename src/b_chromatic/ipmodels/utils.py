def get_total_colouring_from_model(V, E, C, x_vars, y_vars):
    """ Retrieve the colouring from the model\n
        V: set of vertices\n
        E: set of edges\n
        C: set of colours\n
        x_vars: vertices variables\n
        y_vars: edges variables
    """
    f = {}
    for u in V:
        for c in C:
            if x_vars[u,c].value() == 1.0:
                f[u] = c
    for u, v in E:
        for c in C:
            if y_vars[u,v,c].value() == 1.0:
                f[(u,v)] = c
    return f

def is_a_proper_total_colouring(G, E, f):
    """ Check if the given total colouring is proper\n
        G: networkx graph\n
        E: set of edges\n
        f: colouring function
    """
    for u, v in E:
        if f[u] == f[v]: return [(u,v), False]
        for w in G.adj[u]:
            if w != v:
                if (u,w) in f and f[(u,w)] == f[(u,v)]: return [(u,w),(u,v),False]
                elif (w,u) in f and f[(w,u)] == f[(u,v)]: return [(w,u),(u,v),False]
        for w in G.adj[v]:
            if w != u:
                if (v,w) in f and f[(v,w)] == f[(u,v)]: return [(v,w), (u,v),False]
                elif (w,v) in f and f[(w,v)] == f[(u,v)]: return [(w,v), (u,v), False]
    return True