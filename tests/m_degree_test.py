import sys
import os


# Add the src/b_chromatic directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/b_chromatic')))
import networkx as nx
import time

#to test
from utils.m_degree import get_total_m_degree

N = [10, 20, 50, 100, 150, 200, 250, 300]  # Different sizes of graphs to test
for n in N:
    G = nx.fast_gnp_random_graph(n=n, p=0.7)  # Generate a random graph
    start_time = time.time()
    m = get_total_m_degree(G)  # Calculate the total m-degree
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Graph with {n} vertices: Total m-degree = {m}, Time taken = {time_taken:.6f} seconds")
