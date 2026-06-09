from utils.b_chromatic_utils import get_m_degree, get_dense_vertices
from pivoted_trees import is_pivoted, color_pivoted_tree
from non_pivoted_trees import good_set, color_non_pivoted_tree

def colour_tree(T):
    """
    Colour a Tree T
    return: maximum degree, m-degree, number of dense vertices, b-chromatic number, 1 if non pivoted; 0 otherwise
    """
    max_degree= max(dict(T.degree()).values())
    m = get_m_degree(T)
    dense_vertices = get_dense_vertices(T, m)
    v = is_pivoted(T, dense_vertices, m)
    if v is False:
        #we colour the tree
        W = good_set(T, dense_vertices, m)
        good_set_mask = {w:False for w in T.nodes()}
        for w in W: good_set_mask[w] = True
        colors = color_non_pivoted_tree(T, W, good_set_mask, m)
        b_chromatic_number = len(set(colors.values()))
        return max_degree, m, len(dense_vertices), b_chromatic_number, 1, colors, W
    else:
        colors = color_pivoted_tree(T, dense_vertices, v, m)
        b_chromatic_number = len(set(colors.values()))
        return max_degree, m, len(dense_vertices), b_chromatic_number, 0, colors, dense_vertices
