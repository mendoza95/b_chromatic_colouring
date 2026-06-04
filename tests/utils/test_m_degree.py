import pytest
import networkx as nx
from src.utils.m_degree import *

# -- HELPERS / FIXTURES --

@pytest.fixture
def build_disconnected_graph():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2),(2,3),(3,1), (4, 5)])
    return G

# -- TESTS for the get_m_degree function --

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

def test_get_m_degree_disconnected_graph(build_disconnected_graph):
    """Test the get_m_degree function with a disconnected graph."""
    G = build_disconnected_graph
    assert get_m_degree(G) == 3

@pytest.mark.parametrize("invalid_input", [
    None,
    123,
    "not a graph",
    [1, 2, 3],
    {1: 2, 3: 4},
    nx.DiGraph(),
    nx.MultiGraph(),
    nx.MultiDiGraph()
])
def test_get_m_degree_invalid_input(invalid_input):
    """Test the get_m_degree function with invalid inputs."""
    with pytest.raises(TypeError):
        get_m_degree(invalid_input)


# -- TESTS for the get_index_m_degree function --

@pytest.mark.parametrize("G, expected",[
    (nx.Graph(), 0),
    (nx.empty_graph(1), 0),
    (nx.path_graph(3), 2),
    (nx.complete_graph(4), 5),
    (nx.star_graph(5), 5)
])
def test_get_index_m_degree(G, expected):
    """Test the get_index_m_degree function on different graphs and edge cases."""
    assert get_index_m_degree(G) == expected

def test_get_index_m_degree_disconnected_graph(build_disconnected_graph):
    """Test the get_index_m_degree function with a disconnected graph."""
    G = build_disconnected_graph
    assert get_index_m_degree(G) == 3

@pytest.mark.parametrize("invalid_input", [
    None,
    123,
    "not a graph",
    [1, 2, 3],
    {1: 2, 3: 4},
    nx.DiGraph(),
    nx.MultiGraph(),
    nx.MultiDiGraph()
])
def test_get_index_m_degree_invalid_input(invalid_input):
    """Test the get_index_m_degree function with invalid inputs."""
    with pytest.raises(TypeError):
        get_index_m_degree(invalid_input)

# -- TESTS for the get_total_m_degree function --
@pytest.mark.parametrize("G, expected",[
    (nx.Graph(), 0),
    (nx.empty_graph(1), 1),
    (nx.path_graph(2), 3),
    (nx.path_graph(3), 3),
    (nx.path_graph(4), 4),
    (nx.path_graph(5), 5),
    (nx.complete_graph(4), 7),
    (nx.star_graph(5), 6)
])
def test_get_total_m_degree(G, expected):
    """Test the get_total_m_degree function on different graphs and edge cases."""
    assert get_total_m_degree(G) == expected

def test_get_total_m_degree_disconnected_graph(build_disconnected_graph):
    """Test the get_total_m_degree function with a disconnected graph."""
    G = build_disconnected_graph
    assert get_total_m_degree(G) == 5

@pytest.mark.parametrize("invalid_input", [
    None,
    123,
    "not a graph",
    [1, 2, 3],
    {1: 2, 3: 4},
    nx.DiGraph(),
    nx.MultiGraph(),
    nx.MultiDiGraph()
])
def test_get_total_m_degree_invalid_input(invalid_input):
    """Test the get_total_m_degree function with invalid inputs."""
    with pytest.raises(TypeError):
        get_total_m_degree(invalid_input)