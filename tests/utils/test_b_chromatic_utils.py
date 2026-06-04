import pytest
import networkx as nx
from src.utils.b_chromatic_utils import *

# BUILDING A DISCONNECTED GRAPH
@pytest.fixture
def disconnected_graph_and_m_degree():
    G = nx.complete_graph(4)
    G.add_edges_from([(4,5),(5,6),(6,4),(4,7)])
    return (G, 4)

@pytest.fixture
def disconnected_graph():
    G = nx.complete_graph(4)
    G.add_edges_from([(4,5),(5,6),(6,4)])
    return G

@pytest.fixture
def paw_b_colouring_example():
    G = nx.Graph()
    G.add_edges_from([(1,2),(2,3),(3,1),(1,4)])
    f = {1:1, 2:2, 3:3, 4:2}
    return G, f

@pytest.fixture
def empty_graph_function():
    G = nx.Graph()
    f = {}
    u = 0
    return G, f, u

# TESTING check_G_instance function
@pytest.mark.parametrize("G",[
    None,
    [],
    'graph',
    1,
    nx.DiGraph(),
    nx.MultiDiGraph(),
    nx.MultiGraph()
])
def test_check_G_instance(G):
    with pytest.raises(TypeError, match="G must be a networkx graph"):
        check_G_instance(G)

# TESTING get_dense_vertices function

## CORNER AND SOME OTHER SPECIAL GRAPHS
@pytest.mark.parametrize("G, m, expected",[
    (nx.Graph(), 0, []),
    (nx.empty_graph(1), 1, [0]),
    (nx.path_graph(3), 2, [0,1,2]),
    (nx.complete_graph(4), 4, [0,1,2,3]),
    (nx.star_graph(5), 2,[0,1,2,3,4,5])
])
def test_get_dense_vertices(G, m, expected):
    assert get_dense_vertices(G, m) == expected

## DISCONNECTED GRAPH
def test_get_dense_vertices_disconnected_graph(disconnected_graph_and_m_degree):
    assert get_dense_vertices(disconnected_graph_and_m_degree[0], 
                              disconnected_graph_and_m_degree[1]) == [0,1,2,3,4]

# TEST get_biggest_cc FUNCTION

## CORNER CASE 1: EMPTY (NULL) GRAPH
def test_get_biggest_cc_null_graph():
    assert get_biggest_cc(nx.Graph()) == False

## CORNER CASE 2: GRAPH WITH NO EDGES AND COMPLETE GRAPH
@pytest.mark.parametrize("G, expected",[
    (nx.empty_graph(5), 1), #we expect the biggest connected component to have one vertex
    (nx.complete_graph(5), 5) #we expect the biggest connected component to have 5 vertices
])
def test_get_biggest_cc_no_edges_complete_graph(G, expected):
    assert get_biggest_cc(G).number_of_nodes() == expected

## CASE: DISCONNECTED GRAPH THAT CONSISTS OF A COMPLETE GRAPH ON 4 VERTICES AND A TRIANGLE
def test_get_biggest_cc_disconnected_graph(disconnected_graph):
    assert get_biggest_cc(disconnected_graph).number_of_nodes() == 4

# TEST is_u_b_chromatic FUNCTION

## CORNER CASE: G IS NULL GRAPH
def test_is_u_b_chromatic_G_null_f_null(empty_graph_function):
    G, f, u = empty_graph_function
    with pytest.raises(ValueError,match="G is the null graph"):
        is_u_b_chromatic(G, f, u)

## CORNER CASE: KEYWORD u DOES NOT EXISTS IN G
def test_is_u_b_chromatic_u_not_in_G(paw_b_colouring_example):
    G, f = paw_b_colouring_example
    u = 5
    with pytest.raises(KeyError, match="vertex u is not in G"):
        is_u_b_chromatic(G, f, u)

## CORNER CASE: KEYWORD u DOES NOT EXISTS IN f
def test_is_u_b_chromatic_u_not_in_f(empty_graph_function):
    G, f, u = empty_graph_function
    v = 1
    G.add_edge(u,v)
    f = {0:1}# only vertex u is coloured
    with pytest.raises(KeyError, match="vertex u is not in f"):
        is_u_b_chromatic(G, f, v)

## CORNER CASE: FUNCTION (dictionary) f is EMPTY
def test_is_u_b_chromatic_f_empty(empty_graph_function):
    G, f, u = empty_graph_function
    G.add_edge(0,1)
    with pytest.raises(ValueError, match="f is empty"):
        is_u_b_chromatic(G, f, u)

## LOGIC CASE: RETURNS TRUE IF A VERTEX IS B-CHROMATIC
def test_is_u_b_chromatic_case_logic_1(paw_b_colouring_example):
    G, f = paw_b_colouring_example
    u = 1
    assert is_u_b_chromatic(G, f, u) == True

## LOGIC CASE: RETURS FALSE IF A VERTEX IS NOT B-CHROMATIC
def test_is_u_b_chromatic_case_logic_2(paw_b_colouring_example):
    G, f = paw_b_colouring_example
    u = 4
    assert is_u_b_chromatic(G, f, u) == False

# TEST are_b_chromatic FUNCTION

## CORNER CASE 1: G IS THE NULL GRAPH
def test_are_b_chromatic_G_null(empty_graph_function):
    G, f, u= empty_graph_function
    with pytest.raises(ValueError, match="G is the null graph"):
        are_b_chromatic(G, f, {})

## CORNER CASE 2: f IS EMPTY
def test_are_b_chromatic_f_empty(empty_graph_function):
    G, f, u = empty_graph_function
    G.add_node(u)
    V = {u}
    with pytest.raises(ValueError, match="f is empty"):
        are_b_chromatic(G, f, V)

## CORNER CASE 3: V IS EMPTY
def test_are_b_chromatic_V_empty(empty_graph_function):
    G, f, u = empty_graph_function
    G.add_node(u)
    f[u] = 1
    with pytest.raises(ValueError, match="V is empty"):
        are_b_chromatic(G, f, {})

## CORNER CASE 4: THERE EXISTS SOME VERTEX u NOT IN G
def test_are_b_chromatic_u_not_in_G(paw_b_colouring_example):
    G, f = paw_b_colouring_example
    V = [1,2,5]
    with pytest.raises(KeyError, match="There is a vertex in V not in G"):
        are_b_chromatic(G, f, V)

## CORNER CASE 5: THERE EXISTS SOME VERTEX u NOT IN F
def test_are_b_chromatic_u_not_in_f(paw_b_colouring_example):
    G, f = paw_b_colouring_example
    V = [1,2,3]
    f.pop(3)
    with pytest.raises(KeyError, match="There is a vertex in V not in f"):
        are_b_chromatic(G, f, V)

## LOGIC CASE: RETURNS TRUE IF EVERY VERTEX IN V IS B-CHROMATIC
def test_are_b_chromatic_logic_1(paw_b_colouring_example):
    G, f = paw_b_colouring_example
    V = [1,2,3]
    assert are_b_chromatic(G, f, V) == True

## LOGIC CASE: RETURNS FALSE IF SOME VERTEX IN V IS NOT B-CHROMATIC
def test_are_b_chromatic_logic_2(paw_b_colouring_example):
    G, f = paw_b_colouring_example
    V = [1,2,4]
    assert are_b_chromatic(G, f, V) == False

# TEST is_proper FUNCTION

## CORNER CASE 1: G IS THE NULL GRAPH
def test_is_proper_G_null(empty_graph_function):
    G, f, u= empty_graph_function
    f = {0:1}
    with pytest.raises(ValueError, match="G is the null graph"):
        is_proper(G, f)
    

## CORNER CASE 2: f IS EMPTY
def test_is_proper_f_empty(empty_graph_function):
    G, f, u= empty_graph_function
    G.add_node(u)
    with pytest.raises(ValueError, match="f is empty"):
        is_proper(G, f)

## LOGIC CASE: RETURNS TRUE IF f CORRESPONDS TO A PROPER COLOURING
def test_is_proper_logic_1(paw_b_colouring_example):
    G, f = paw_b_colouring_example
    assert is_proper(G, f) == True

## LOGIC CASE: RETURNS FALSE IF f DOES NOT CORRESPOND TO A PROPER COLOURING
def test_is_proper_logic_2(paw_b_colouring_example):
    G, f = paw_b_colouring_example
    f[4] = 1# we change the colour of vertex 4 to be the same as vertex 1
    assert is_proper(G, f) == False

# TEST get_b_chromatic_vertices FUNCTION

## CORNER CASE 1: G IS THE NULL GRAPH
def test_get_b_chromatic_vertices_G_null(empty_graph_function):
    G, f, u = empty_graph_function
    with pytest.raises(ValueError, match="G is the null graph"):
        get_b_chromatic_vertices(G, f)

## CORNER CASE 2: f IS EMPTY
def test_get_b_chromatic_vertices_f_empty(empty_graph_function):
    G, f, u = empty_graph_function
    G.add_node(u)
    with pytest.raises(ValueError, match="f is empty"):
        get_b_chromatic_vertices(G, f)

## LOGIC CASE: RETURNS A DICTIONARY WHERE THE KEY ARE COLOUR CLASSES AND THE VALUES THE LIST OF B-CHROMATIC VERTICES OF THEIR KEY
def test_get_b_chromatic_vertices_logic(paw_b_colouring_example):
    G, f = paw_b_colouring_example
    G.add_edge(3,4)
    assert get_b_chromatic_vertices(G, f) == {1:[1], 2:[2,4], 3:[3]}