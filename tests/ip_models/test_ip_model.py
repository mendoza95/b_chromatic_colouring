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
from ip_models.total_b_chromatic_colouring.total_b_chromatic_model import find_total_b_chromatic_colouring


MAX_TIME_LIMIT = 100
D = range(10)
N = [20]
resultspd = {"d":[], "V": [], "E": [], "m(G)": [], "sol_status": [], "b(G)":[], "time": []}
for d in D:
    print("Execution {}".format(d+1))
    for n in N:
        G = nx.fast_gnp_random_graph(n=n,p=0.7)
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

now = datetime.now()
filename = "../b_chromatic_files/test_results/total_b_chromatic_colouring_ip_models/10_random_graph_with_20_vertices_{}_{}_{}_{}_{}.csv"
pd.DataFrame(resultspd).to_csv(filename.format(now.day, now.month, now.year, now.hour, now.minute))