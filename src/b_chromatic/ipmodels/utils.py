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

def get_total_b_chromatic_elements(V, E, C, t_vars, z_vars):
    """ Get the total b-chromatic elements from the model\n
        V: set of vertices\n
        E: set of edges\n
        C: set of colours\n
        t_vars: total b-chromatic vertex variables\n
        z_vars: total b-chromatic edge variables
    """
    b_chromatic_elements = {}
    for u in V:
        for c in C:
            if t_vars[u,c].value() == 1.0:
                b_chromatic_elements[c] = u
    for u, v in E:
        for c in C:
            if z_vars[u,v,c].value() == 1.0:
                b_chromatic_elements[c] = (u,v)
    return b_chromatic_elements


def is_a_total_b_chromatic_colouring(G, V, E, f, bchr_elements):
    """ Check if the total colouring is a total b-chromatic colouring\n
        G: networkx graph\n
        V: set of vertices\n
        E: set of edges\n
        f: total colouring function\n
        bchr_elements: total b-chromatic elements

    """
    m = len(set(f.values()))
    for e in bchr_elements.values():
        Cp = set()
        if e in V:
            for w in G.adj[e]:
                Cp.add(f[w])
                if (w,e) in E: Cp.add(f[w,e])
                else: Cp.add(f[e,w])
        else:
            u = e[0]; v = e[1]
            Cp.add(f[u]); Cp.add(f[v])
            for w in G.adj[u]:
                if w != v:
                    if (w,u) in E: Cp.add(f[w,u])
                    else: Cp.add(f[u,w])
            for w in G.adj[v]:
                if w != u:
                    if (w,v) in E: Cp.add(f[w,v])
                    else: Cp.add(f[v,w])
        if len(Cp) != m-1: return False, e, Cp
    return True
