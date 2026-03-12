import math

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

def validate_full_solution(V, E, C, variables):
    """
    Asserts that the raw solver output respects all major integrity and logic constraints.
    """
    w_vars = variables['w']
    x_vars = variables['x']
    y_vars = variables['y']
    t_vars = variables['t']
    z_vars = variables['z']

    # --- Integrity Checks (formerly validate_solution_integrity) ---
    # Assert that each vertex has exactly one color assigned
    for u in V:
        sum_x_u = sum(x_vars[u, c].value() for c in C)
        assert math.isclose(sum_x_u, 1.0), \
            f"Assertion failed: Vertex {u} does not have exactly one color assigned. Sum was {sum_x_u}"

    # Assert that each edge has exactly one color assigned
    for u, v in E:
        sum_y_uv = sum(y_vars[u, v, c].value() for c in C)
        assert math.isclose(sum_y_uv, 1.0), \
            f"Assertion failed: Edge ({u},{v}) does not have exactly one color assigned. Sum was {sum_y_uv}"

    # --- B-Chromatic Property Checks (formerly validate_b_chromatic_properties) ---
    for j in C:
        # Check if w_j is consistent with color assignments
        is_color_j_used = any(x_vars[u, j].value() == 1.0 for u in V) or \
                          any(y_vars[u, v, j].value() == 1.0 for u, v in E)

        if math.isclose(w_vars[j].value(), 1.0):
            assert is_color_j_used, f"Assertion failed: w_{j}=1 but no vertex or edge uses color {j}."

            # Check if a b-chromatic element exists for a used color
            has_b_element = any(t_vars[u, j].value() == 1.0 for u in V) or \
                            any(z_vars[u, v, j].value() == 1.0 for u, v in E)
            assert has_b_element, f"Assertion failed: w_{j}=1 but no b-chromatic element for color {j} exists."

        else: # w_j is 0
            assert not is_color_j_used, f"Assertion failed: w_{j}=0 but some vertex or edge uses color {j}."

        # Check sequential color usage
        if j > 0:
            assert w_vars[j].value() <= w_vars[j-1].value(), f"Assertion failed: Color {j} is used but color {j-1} is not (sequential constraint violation)."

    # Check t_vars consistency
    for u in V:
        for j in C:
            assert t_vars[u, j].value() <= x_vars[u, j].value(), f"Assertion failed: t_{u},{j}=1 but x_{u},{j}=0. A vertex must have a color to be a b-vertex for it."

    # Check z_vars consistency
    for u, v in E:
        for j in C:
            assert z_vars[u, v, j].value() <= y_vars[u, v, j].value(), f"Assertion failed: z_{u},{v},{j}=1 but y_{u},{v},{j}=0. An edge must have a color to be a b-edge for it."
