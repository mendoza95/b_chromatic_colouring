import pulp as pl
import networkx as nx
from m_degree import get_total_m_degree


#VARIABLES
def get_colour_var(C):
    """C: set of colours"""
    w_vars = {j:pl.LpVariable(cat=pl.LpBinary,name="w_{0}".format(j)) 
                        for j in C}
    return w_vars

def get_vertex_colour_var(V, C):
    """ V: set of vertices\n
        C: set of colours"""
    x_vars = {(u,j):pl.LpVariable(cat=pl.LpBinary,name="x_{0}_{1}".format(u,j))
          for u in V for j in C}
    return x_vars

def get_edge_colour_var(E, C):
    """E: set of edges\n
       C: set of colours"""
    y_vars = {(u,v,j):pl.LpVariable(cat=pl.LpBinary,name="y_{0}_{1}_{2}".format(u,v,j))
            for u,v in E for j in C}
    return y_vars

def get_b_chromatic_vertex_var(V, C):
    """ V: set of vertices\n
        C: set of colours"""
    t_vars = {(u,j):pl.LpVariable(cat=pl.LpBinary,name="t_{0}_{1}".format(u,j))
            for u in V for j in C}
    return t_vars

def get_b_chromatic_edge_var(E, C):
    """ E: set of edges\n
        C: set of colours"""
    z_vars = {(u,v,j):pl.LpVariable(cat=pl.LpBinary,name="z_{0}_{1}_{2}".format(u,v,j))
            for u,v in E for j in C}
    return z_vars


#CONSTRAINTS

##ONE COLOUR PER OBJECT

def add_one_colour_per_vertex_constraint(model, x_vars, V, C):
    """ Every vertex must have only one colour assigned to it\n
        model: IP model\n
        x_vars: vertices variables\n
        V: set of vertices\n
        C: set of colours
    """
    one_colour_per_vertex = {u:model.addConstraint(
        pl.LpConstraint(
            e = pl.lpSum(x_vars[u,j] for j in C),
            sense=pl.LpConstraintEQ,
            rhs=1,
            name="one_colour_vertex_{0}".format(u)
        )
    )
    for u in V}
    return model

def add_one_colour_per_edge_constraint(model, y_vars, E, C):
    """ Every edge must have only one colour assigned to it\n
        model: IP model\n
        y_vars: edges variables\n
        E: set of edges\n
        C: set of colours
    """
    one_colour_per_edge = {(u,v):model.addConstraint(
        pl.LpConstraint(
            e = pl.lpSum(y_vars[u,v,j] for j in C),
            sense=pl.LpConstraintEQ,
            rhs=1,
            name="one_colour_edge_{0}_{1}".format(u,v)
        )
    )
    for u, v in E}
    return model


##DIFFERENT COLOURS PER ADJACENT OBJECTS


def add_different_colour_adj_vertices_constraint(model, x_vars, E, C):
    """ Every pair of adjacent vertices must have different colours\n
        model: IP model\n
        x_vars: vertices variables\n
        E: set of edges\n
        C: set of colours    
    """
    different_colour_adj_vertices = {(u,v,j):model.addConstraint(
        pl.LpConstraint(
            e = x_vars[u,j]+x_vars[v,j],
            sense=pl.LpConstraintLE,
            rhs=1,
            name="different_colour_adj_vertices_{0}_{1}_{2}".format(u,v,j)
        )
    )
    for u,v in E for j in C}
    return model

def add_different_colour_adj_edges_constraint(model, y_vars, E, C):
    """ Every pair of adjacent edges must have different colours\n
        model: IP model\n
        y_vars: edges variables\n
        E: set of edges\n
        C: set of colours    
    """
    different_colour_adj_edges = {(u1,v1,u2,v2,j):model.addConstraint(
        pl.LpConstraint(
            e = y_vars[u1,v1,j]+y_vars[u2,v2,j],
            sense=pl.LpConstraintLE,
            rhs=1,
            name="different_colour_adj_vertices_{0}_{1}_{2}_{3}_{4}".format(u1,v1,u2,v2,j)
        )
    )
    for i, (u1,v1) in enumerate(E) for u2,v2 in E[i+1:] 
    for j in C
    if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2}
    return model

def add_different_colour_vertex_incident_edge_constraint(model, x_vars, y_vars, V, E, C):
    """ Every pair adjacent vertex and edge must have different colours\n
        model: IP model\n
        x_vars: vertices variables\n
        y_vars: edges variables\n
        V: set of vertices\n
        E: set of edges\n
        C: set of colours
    """
    different_colour_vertex_incident_edge = {(w,u,v,j):model.addConstraint(
        pl.LpConstraint(
            e = x_vars[w,j]+y_vars[u,v,j],
            sense=pl.LpConstraintLE,
            rhs=1,
            name="different_colour_vertex_{0}_incident_edge_{1}_{2}_colour_{3}".format(w,u,v,j)
        )
    )
    for w in V for u,v in E for j in C if w == u or w == v}
    return model

## $w_j=1$ IF AND ONLY IF COLOUR j WAS ASSIGNED TO SOME EDGE OR VERTEX

### (==>)

def add_colour_assigned_to_some_vertex_if_constraint(model, x_vars, w_vars, V, C):
    """ If colour j is asigned to some vertex then x_{uj}\leq w_j for every vertex $u\in V$\n
        model: IP model\n
        x_vars: vertices variables\n
        w_vars: colour variables\n
        V: set of vertices\n
        C: set of colours
    """
    colour_assigned_to_some_vertex_if = {(u,j):model.addConstraint(
        pl.LpConstraint(
            e=x_vars[u,j]-w_vars[j],
            sense=pl.LpConstraintLE,
            rhs=0,
            name="colour_{0}_assigned_to_some_vertex_{1}_if".format(u,j)
        )
    )
    for u in V for j in C}
    return model

def add_colour_assigned_to_some_edge_if_constraint(model, y_vars, w_vars, E, C):
    """ If colour j is asigned to some edge then e_{uj}\leq w_j for every vertex $e\in E$\n
        model: IP model\n
        y_vars: edges variables\n
        w_vars: colour variables\n
        E: set of edges\n
        C: set of colours
    """
    colour_assigned_to_some_edge_if = {(u,v,j):model.addConstraint(
        pl.LpConstraint(
            e=y_vars[u,v,j]-w_vars[j],
            sense=pl.LpConstraintLE,
            rhs=0,
            name="colour_{0}_assigned_to_some_edge_{1}_{2}_if".format(u,v,j)
        )
    )
    for u,v in E for j in C}
    return model

### (<==)

def add_colour_assigned_to_some_vertex_edge_onlyif_constraint(model, x_vars, y_vars,w_vars, V, E, C):
    """ If colour j is assigned to some vertex or edge then $\sum_{u\in V}x_{uj} + \sum_{e\in E}y_{ej}\geq w_j$\n 
        model: IP model\n
        x_vars: vertices variables\n
        y_vars: edges variables\n
        w_vars: colour variables\n
        V: set of vertices\n
        E: set of edges\n
        C: set of colours
    """
    colour_assigned_to_some_vertex_edge_onlyif = {j:model.addConstraint(
        pl.LpConstraint(
            e=pl.lpSum(x_vars[u,j] for u in V)+pl.lpSum(y_vars[u,v,j] for u,v in E)-w_vars[j],
            sense=pl.LpConstraintGE,
            rhs=0,
            name="colour_{0}_assigned_to_some_vertex_edge_onlyif".format(j)
        )
    ) for j in C}
    return model

# TOTAL B-CHROMATIC COLOURING CONSTRAINTS

def add_total_b_chromatic_vertex_colour_j_constraint(model, t_vars, w_vars, x_vars, y_vars, V, E, C, G):
    """ If there u is a total b-chromatic vertex of colour j
        model: IP model\n
        t_vars: total b-chromatic vertices variables\n
        w_vars: colour variables\n
        x_vars: vertices variables\n
        y_vars: edges variables\n
        V: set of vertices\n
        E: set of edges\n
        C: set of colours\n
        G: networkx graph
    """
    total_b_chromatic_vertex_colour_j = {(u,j, jp): model.addConstraint(
        pl.LpConstraint(
            e = t_vars[u,j]+w_vars[jp]-pl.lpSum(x_vars[v,jp] for v in G.adj[u])-pl.lpSum(y_vars[a,b,jp] for a, b in E if a == u or b == u),
            sense=pl.LpConstraintLE,
            rhs=1,
            name="total_b_chromatic_vertex_{0}_colour_{1}_adj_inc_element_colour_{2}".format(u,j,jp)
        )
    )
    for u in V for j in C for jp in C if j != jp}
    return model

def add_total_b_chromatic_edge_colour_j_constraint(model, z_vars, w_vars, y_vars, x_vars, E, C):
    """ If there e is a total b-chromatic edge of colour j
        model: IP model\n
        z_vars: total b-chromatic edges variables\n
        w_vars: colour variables\n
        y_vars: edges variables\n
        x_vars: vertices variables\n
        E: set of edges\n
        C: set of colours\n
    """
    total_b_chromatic_edge_colour_j = {(u,v,j,jp): model.addConstraint(
        pl.LpConstraint(
            e = z_vars[u,v,j]+w_vars[jp]
                             -pl.lpSum(y_vars[u1,v1,jp] for u1, v1 in E if len({u,v}.intersection({u1,v1})) == 1)
                             -x_vars[u,jp]-x_vars[v,jp],
            sense=pl.LpConstraintLE,
            rhs=1,
            name="total_b_chromatic_edge_{0}_{1}_colour_{2}_adj_inc_element_colour_{3}".format(u,v,j,jp)
        )
    )
    for u, v in E for j in C for jp in C if j != jp}
    return model

def add_there_is_total_b_chromatic_element_for_every_colour_constraint(model, t_vars, z_vars, w_vars, V, E, C):
    """ There is a total b-chromatic element for every colour used\n
        t_vars: total b-chromatic vertices variables\n
        z_vars: total b-chromatic edges variables\n
        w_vars: colour variables\n
        V: set of vertices\n
        E: set of edges\n
        C: set of colours
    """
    there_is_total_b_chromatic_element_for_every_colour = {j:model.addConstraint(
        pl.LpConstraint(
            e= pl.lpSum(t_vars[u,j] for u in V)
              +pl.lpSum(z_vars[u,v,j] for u,v in E)
              -w_vars[j],
            sense=pl.LpConstraintGE,
            rhs=0,
            name="there_is_total_b_chromatic_element_for_colour_{0}".format(j)
        )
    ) 
    for j in C}
    return model

## USE ALL THE COLOURS SEQUENTIALLY

def add_use_all_colours_sequentially_constraint(model, w_vars, C):
    """ Use all the colours sequentially\n
        w_vars: colour variables\n
        C: set of colours
    """
    use_all_colours_sequentially = {(j,jp):model.addConstraint(
        pl.LpConstraint(
            e=w_vars[j]-w_vars[jp],
            sense=pl.LpConstraintLE,
            rhs=0,
            name="use_colour_{0}_before_{1}".format(j,jp)
        )
    )
    for j in C for jp in C if jp < j}
    return model

#OBJECTIVE FUNCTION

def set_objective_function(model, sense, obj_vars, S):
    """ Maximamise the number of colours used\n
        sense: MAX or MIN\n
        obj_vars: objective variables\n
        S: objective variables indices
    """
    total_b_chromatic_number = pl.lpSum(obj_vars[j] for j in S)
    model.sense = sense
    model.setObjective(total_b_chromatic_number)
    return model


# BUILDING THE MODEL

def get_total_b_chromatic_model(G, V, E, m):
    """ Construct a IP model for graph G\n
        G: networkx graph\n
        V: set of vertices\n
        E: set of edges\n
        m: total m-degree of G
    """
    b_chr_total_model = pl.LpProblem(name="b-chromatic-total-model")
    C = range(m)

    # SETTING VARIABLES
    w_vars = get_colour_var(C)
    x_vars = get_vertex_colour_var(V, C)
    y_vars = get_edge_colour_var(E, C)
    t_vars = get_b_chromatic_vertex_var(V, C)
    z_vars = get_b_chromatic_edge_var(E, C)

    ## TOTAL COLOURING CONSTRAINTS
    b_chr_total_model = add_one_colour_per_vertex_constraint(b_chr_total_model, x_vars, V, C)
    b_chr_total_model = add_one_colour_per_edge_constraint(b_chr_total_model, y_vars, E, C)
    b_chr_total_model = add_different_colour_adj_vertices_constraint(b_chr_total_model, x_vars, E, C)
    b_chr_total_model = add_different_colour_adj_edges_constraint(b_chr_total_model, y_vars, E, C)
    b_chr_total_model = add_different_colour_vertex_incident_edge_constraint(b_chr_total_model, x_vars, y_vars, V, E, C)

    #COLOUR ASSIGNED TO SOME VERTEX OF EDGE CONSTRAINT
    b_chr_total_model = add_colour_assigned_to_some_vertex_if_constraint(b_chr_total_model, x_vars, w_vars, V, C)
    b_chr_total_model = add_colour_assigned_to_some_edge_if_constraint(b_chr_total_model, y_vars, w_vars, E, C)

    b_chr_total_model = add_colour_assigned_to_some_vertex_edge_onlyif_constraint(b_chr_total_model, x_vars, y_vars, w_vars, V, E, C)

    #B-CHROMATIC COLOURING CONSTRAINTS

    b_chr_total_model = add_total_b_chromatic_vertex_colour_j_constraint(b_chr_total_model, t_vars, w_vars, x_vars, y_vars, V, E, C,G)
    b_chr_total_model = add_total_b_chromatic_edge_colour_j_constraint(b_chr_total_model, z_vars, w_vars, y_vars, x_vars, E, C)

    b_chr_total_model = add_there_is_total_b_chromatic_element_for_every_colour_constraint(b_chr_total_model, t_vars, z_vars, w_vars, V, E, C)


    #SIMETRY CONSTRAINT
    b_chr_total_model = add_use_all_colours_sequentially_constraint(b_chr_total_model,w_vars, C)

    #OBJECTIVE FUNCTION
    b_chr_total_model = set_objective_function(b_chr_total_model, pl.LpMaximize, w_vars, C)

    return b_chr_total_model