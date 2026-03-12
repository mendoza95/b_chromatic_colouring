#importing src/b_chromtic files
import sys, os
# Add the src/b_chromatic directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/b_chromatic')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/b_chromatic/ip_models')))
from datetime import datetime

import networkx as nx
import pandas as pd

from utils.m_degree import get_total_m_degree

#to test
from ip_models.total_b_chromatic_colouring.total_b_chromatic_model import get_total_b_chromatic_model



MAX_TIME_LIMIT = 100
N = [10, 15, 20, 25, 30, 35, 40] # number of vertices
results = {"N": [], "E":[], "time_taken": [], "# variables": [], "# constraints": [], "# non-zero coefficients": []}
for n in N:
    G = nx.fast_gnp_random_graph(n=n,p=0.7)
    V = list(G.nodes())
    E = list(G.edges())
    m = get_total_m_degree(G)
    start_time = datetime.now()
    b_chr_total_model, _, _, _, _, _ = get_total_b_chromatic_model(G, V, E, m)
    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()
    results["N"].append(n)
    results["E"].append(len(E))
    results["time_taken"].append(time_taken)
    results["# variables"].append(len(b_chr_total_model.variables()))
    results["# constraints"].append(len(b_chr_total_model.constraints))
    # Count non-zero coefficients
    non_zero_coeff = 0
    for constraint in b_chr_total_model.constraints.values():
        non_zero_coeff += sum(1 for v in constraint.items() if v[1] != 0)
    results["# non-zero coefficients"].append(non_zero_coeff)
    print(f"Model built in {time_taken} seconds.")

now = datetime.now()
filename = "../../b_chromatic_files/test_results/ip_model/model_build/test_on_{}_{}_{}_{}_{}.csv"
pd.DataFrame(results).to_csv(filename.format(now.day, now.month, now.year, now.hour, now.minute))


