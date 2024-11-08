#importing src/b_chromtic files
import sys, os
sys.path.append('../src/b_chromatic')
sys.path.append('../src/b_chromatic/ipmodels')
#pandas
import pandas as pd
import networkx as nx

#to test
from ipmodels.total_b_chromatic_model import find_total_b_chromatic_colouring
from m_degree import get_total_m_degree


MAX_TIME_LIMIT = 100
D = range(3, 4)
N = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
resultspd = {"d":[], "V": [], "E": [], "m(G)": [], "sol_status": [], "b(G)":[], "time": []}
for d in D:
    for n in N:
        G = nx.random_regular_graph(d=d,n=n)
        V = list(G.nodes())
        E = list(G.edges())
        m = get_total_m_degree(G)
        sol_status, obj_value, time_taken, f, bchr_elements = find_total_b_chromatic_colouring(G, V, E, m, MAX_TIME_LIMIT)
        resultspd["d"].append(d)
        resultspd["V"].append(n)
        resultspd["E"].append(len(E))
        resultspd["m(G)"].append(m)
        resultspd["sol_status"].append(sol_status)
        resultspd["b(G)"].append(obj_value)
        resultspd["time"].append(time_taken)

pd.DataFrame(resultspd).to_csv("../b_chromatic_files/test_results/total_b_chromatic_colouring_ip_models/d_regular_graphs.csv")