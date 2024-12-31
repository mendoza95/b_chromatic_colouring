import sys
sys.path.append("..")
import pulp as pl
import networkx as nx
from variables import *
from constraints import *
from utils import *
from m_degree import get_total_m_degree



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
    #b_chr_total_model = add_colour_assigned_to_some_vertex_if_constraint(b_chr_total_model, x_vars, w_vars, V, C)
    #b_chr_total_model = add_colour_assigned_to_some_edge_if_constraint(b_chr_total_model, y_vars, w_vars, E, C)

    b_chr_total_model = add_colour_assigned_to_some_vertex_edge_onlyif_constraint(b_chr_total_model, x_vars, y_vars, w_vars, V, E, C)

    #B-CHROMATIC COLOURING CONSTRAINTS

    b_chr_total_model = add_total_b_chromatic_vertex_colour_j_constraint(b_chr_total_model, t_vars, w_vars, x_vars, y_vars, V, E, C,G)
    b_chr_total_model = add_total_b_chromatic_edge_colour_j_constraint(b_chr_total_model, z_vars, w_vars, y_vars, x_vars, E, C)

    b_chr_total_model = add_there_is_total_b_chromatic_element_for_every_colour_constraint(b_chr_total_model, t_vars, z_vars, w_vars, V, E, C)


    #SIMETRY CONSTRAINT
    b_chr_total_model = add_use_all_colours_sequentially_constraint(b_chr_total_model,w_vars, C)

    #OBJECTIVE FUNCTION
    b_chr_total_model = set_objective_function(b_chr_total_model, pl.LpMaximize, w_vars, C)

    return b_chr_total_model, x_vars, y_vars, w_vars, t_vars, z_vars

def find_total_b_chromatic_colouring(G, V, E, m, time_limit):
    """ Constructs the integer programming model for total b-chromatic colouring\n
        G: networkx graph\n
        V: set of vertices\n
        E: set of edges\n
        time_limit: max allowed time to find a solution\n
        Return: solution status, objective value, solution time, total colouring (if found), total b-chromatic elements (if found)
    """
    C = range(m)
    bchr_model,  x_vars, y_vars, w_vars, t_vars, z_vars = get_total_b_chromatic_model(G, V, E, m)
    bchr_model.solve(pl.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    f = get_total_colouring_from_model(V, E, C, x_vars, y_vars)
    b_chr_elements = get_total_b_chromatic_elements(V, E, C, t_vars, z_vars)
    if bchr_model.sol_status == 1:
        if is_a_proper_total_colouring(G, E, f) and is_a_total_b_chromatic_colouring(G, V, E, f, b_chr_elements):
            return bchr_model.sol_status, bchr_model.objective.value(), bchr_model.solutionTime, f, b_chr_elements
    return bchr_model.sol_status, None, bchr_model.solutionTime, None, None
        

if __name__ == "__main__":
    G = nx.random_regular_graph(d=3,n=10)
    V = list(G.nodes())
    E = list(G.edges())
    m = get_total_m_degree(G)
    print(find_total_b_chromatic_colouring(G, V, E, m, 10))
    