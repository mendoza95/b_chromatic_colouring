#importing src/b_chromtic files
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/b_chromatic')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/b_chromatic/ip_models')))
from datetime import datetime
#pandas
import pandas as pd
import networkx as nx
import pulp as pl

from utils.m_degree import get_total_m_degree
from ip_models.total_b_chromatic_colouring.variables import get_colour_var, get_vertex_colour_var, get_edge_colour_var, \
    get_b_chromatic_vertex_var, get_b_chromatic_edge_var

#to test
from ip_models.total_b_chromatic_colouring.constraints import add_one_colour_per_vertex_constraint, \
    add_one_colour_per_edge_constraint, \
    add_different_colour_adj_vertices_constraint, \
    add_different_colour_adj_edges_constraint, \
    add_different_colour_vertex_incident_edge_constraint, \
    add_colour_assigned_to_some_vertex_if_constraint, \
    add_colour_assigned_to_some_edge_if_constraint,\
    add_colour_assigned_to_some_vertex_edge_onlyif_constraint,\
    add_total_b_chromatic_vertex_colour_j_constraint,\
    add_total_b_chromatic_edge_colour_j_constraint,\
    add_there_is_total_b_chromatic_element_for_every_colour_constraint,\
    add_use_all_colours_sequentially_constraint

N = [10, 15, 20, 25, 30, 35, 40]  # number of vertices
results = {
    "N": [],
    "E": [],
    "add_one_colour_per_vertex_constraint": [],
    "add_one_colour_per_edge_constraint": [],
    "add_different_colour_adj_vertices_constraint": [],
    "add_different_colour_adj_edges_constraint": [],
    "add_different_colour_vertex_incident_edge_constraint": [],
    "add_colour_assigned_to_some_vertex_edge_onlyif_constraint": [],
    "add_total_b_chromatic_vertex_colour_j_constraint": [],
    "add_total_b_chromatic_edge_colour_j_constraint": [],
    "add_there_is_total_b_chromatic_element_for_every_colour_constraint": [],
    "add_use_all_colours_sequentially_constraint": []
}
for n in N:
    G = nx.fast_gnp_random_graph(n=n, p=0.7)
    V = list(G.nodes())
    E = list(G.edges())
    m = get_total_m_degree(G)

    results["N"].append(n)
    results["E"].append(len(E))

    b_chr_total_model = pl.LpProblem(name="b-chromatic-total-model")
    C = range(m)

    # SETTING VARIABLES
    w_vars = get_colour_var(C)
    x_vars = get_vertex_colour_var(V, C)
    y_vars = get_edge_colour_var(E, C)
    t_vars = get_b_chromatic_vertex_var(V, C)
    z_vars = get_b_chromatic_edge_var(E, C)

    # Test each constraint
    start_time = datetime.now()
    model = add_one_colour_per_vertex_constraint(b_chr_total_model, x_vars, V, C)
    results["add_one_colour_per_vertex_constraint"].append((datetime.now() - start_time).total_seconds())

    start_time = datetime.now()
    model = add_one_colour_per_edge_constraint(b_chr_total_model, y_vars, E, C)
    results["add_one_colour_per_edge_constraint"].append((datetime.now() - start_time).total_seconds())

    start_time = datetime.now()
    model = add_different_colour_adj_vertices_constraint(b_chr_total_model, x_vars, E, C)
    results["add_different_colour_adj_vertices_constraint"].append((datetime.now() - start_time).total_seconds())

    start_time = datetime.now()
    model = add_different_colour_adj_edges_constraint(b_chr_total_model, y_vars, E, C)
    results["add_different_colour_adj_edges_constraint"].append((datetime.now() - start_time).total_seconds())

    start_time = datetime.now()
    model = add_different_colour_vertex_incident_edge_constraint(b_chr_total_model, x_vars, y_vars, V, E, C)
    results["add_different_colour_vertex_incident_edge_constraint"].append((datetime.now() - start_time).total_seconds())

    start_time = datetime.now()
    model = add_colour_assigned_to_some_vertex_edge_onlyif_constraint(b_chr_total_model, x_vars, y_vars, w_vars, V, E, C)
    results["add_colour_assigned_to_some_vertex_edge_onlyif_constraint"].append((datetime.now() - start_time).total_seconds())
    
    start_time = datetime.now()
    model = add_total_b_chromatic_vertex_colour_j_constraint(b_chr_total_model, t_vars, w_vars, x_vars, y_vars, V, E, C,G)
    results["add_total_b_chromatic_vertex_colour_j_constraint"].append((datetime.now() - start_time).total_seconds())

    start_time = datetime.now()
    model = add_total_b_chromatic_edge_colour_j_constraint(b_chr_total_model, z_vars, w_vars, y_vars, x_vars, E, C)
    results["add_total_b_chromatic_edge_colour_j_constraint"].append((datetime.now() - start_time).total_seconds())

    start_time = datetime.now()
    model = add_there_is_total_b_chromatic_element_for_every_colour_constraint(b_chr_total_model, t_vars, z_vars, w_vars, V, E, C)
    results["add_there_is_total_b_chromatic_element_for_every_colour_constraint"].append((datetime.now() - start_time).total_seconds())

    start_time = datetime.now()
    model = add_use_all_colours_sequentially_constraint(b_chr_total_model,w_vars, C)
    results["add_use_all_colours_sequentially_constraint"].append((datetime.now() - start_time).total_seconds())



now = datetime.now()
filename = "../../b_chromatic_files/test_results/ip_model/constraints_build/test_on_{}_{}_{}_{}_{}.csv"
pd.DataFrame(results).to_csv(filename.format(now.day, now.month, now.year, now.hour, now.minute))