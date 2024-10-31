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
    """ Every vertex must have only one colour assigned to it
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
    """ Every edge must have only one colour assigned to it
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
    """ Every pair of adjacent vertices must have different colours
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

#OBJECTIVE FUNCTION
