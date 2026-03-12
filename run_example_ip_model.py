import networkx as nx
from src.b_chromatic.utils.m_degree import get_total_m_degree
from src.b_chromatic.ip_models.total_b_chromatic_colouring.total_b_chromatic_model import find_total_b_chromatic_colouring

if __name__ == "__main__":
    G = nx.Graph()
    V = [1, 2, 3, 4, 5, 6]
    E = [(5,2), (5,1), (1,2), (2,3), (3,4), (4,1), (2,4), (4,6), (3,6)]
    G.add_nodes_from(V)
    G.add_edges_from(E)
    m = get_total_m_degree(G)
    print("Finding total b-chromatic colouring...")
    sol_status, obj_value, sol_time, total_colouring,  total_b_chr_elements = find_total_b_chromatic_colouring(G, V, E, m, time_limit=10)
    print("Solution Status:", sol_status)
    print("Objective Value (number of colours used):", obj_value)
    print("Solution Time (seconds):", sol_time)
    
    print("Total Colouring (if found):")
    if total_colouring:
        for element, colour in total_colouring.items():
            print(f"  Element {element}: Colour {colour}")
    else:
        print("  None")

    print("Total b-Chromatic Elements (if found):")
    if total_b_chr_elements:
        for colour, element in total_b_chr_elements.items():
            print(f"  Element {element}: Colour {colour}")
    else:
        print("  None")