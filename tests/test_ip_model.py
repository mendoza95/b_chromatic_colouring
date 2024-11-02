import sys, os
import networkx as nx
import pulp as pl
#importing src/b_chromtic files
sys.path.append('../src/b_chromatic')
sys.path.append('../src/b_chromatic/ipmodels')
from ipmodels.total_b_chromatic_model import get_total_b_chromatic_model
from m_degree import get_total_m_degree



G = nx.random_regular_graph(d=3,n=10)
V = list(G.nodes())
E = list(G.edges())
m = get_total_m_degree(G)
b_chr_model,  x_vars, y_vars, w_vars, t_vars, z_vars = get_total_b_chromatic_model(G, V, E, m)
b_chr_model.solve(pl.PULP_CBC_CMD(msg=1))
