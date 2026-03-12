#importing src/b_chromtic files
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/b_chromatic')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/b_chromatic/ip_models')))
from datetime import datetime
#pandas
import pandas as pd
import networkx as nx

from utils.m_degree import get_total_m_degree

#to test
from ip_models.total_b_chromatic_colouring.variables import get_colour_var, get_vertex_colour_var, get_edge_colour_var, get_b_chromatic_vertex_var, get_b_chromatic_edge_var

N = [10, 15, 20, 25, 30, 35, 40] # number of vertices
results = {"N": [], "E":[], "w-vars": [], "x-vars": [], "y-vars": [], "z-vars": [], "t-vars": []}
for n in N:
    G = nx.fast_gnp_random_graph(n=n,p=0.7)
    V = list(G.nodes())
    E = list(G.edges())
    m = get_total_m_degree(G)

    results["N"].append(n)

    results["E"].append(len(E))
    
    start_time = datetime.now()
    w_vars = get_colour_var(range(m))
    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()
    results["w-vars"].append(time_taken)

    start_time = datetime.now()
    x_vars = get_vertex_colour_var(V, range(m))
    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()
    results["x-vars"].append(time_taken)

    start_time = datetime.now()
    y_vars = get_edge_colour_var(E, range(m))
    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()
    results["y-vars"].append(time_taken)

    start_time = datetime.now()
    t_vars = get_b_chromatic_vertex_var(V, range(m))
    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()
    results["t-vars"].append(time_taken)

    start_time = datetime.now()
    z_vars = get_b_chromatic_edge_var(E, range(m))
    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()
    results["z-vars"].append(time_taken)

now = datetime.now()
filename = "../../b_chromatic_files/test_results/ip_model/variables_build/test_on_{}_{}_{}_{}_{}.csv"
pd.DataFrame(results).to_csv(filename.format(now.day, now.month, now.year, now.hour, now.minute))
    