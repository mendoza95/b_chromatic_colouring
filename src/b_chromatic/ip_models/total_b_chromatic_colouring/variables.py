import pulp as pl

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