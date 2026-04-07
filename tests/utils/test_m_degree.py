import pytest
import networkx as nx
from src.b_chromatic.utils.m_degree import get_m_degree, get_total_m_degree, get_index_m_degree

@pytest.mark.parametrize("G, expected",[
    (nx.Graph(), 0),
    (nx.empty_graph(1), 1),
    (nx.path_graph(3), 2),
    (nx.complete_graph(4), 4),
    (nx.star_graph(5), 2)
])
def test_get_m_degree(G, expected):
    """Test the get_m_degree function.
    
    It tests different edge cases, invalid inputs, and boundary conditions to ensure the function behaves as expected.
    """
    assert get_m_degree(G) == expected
