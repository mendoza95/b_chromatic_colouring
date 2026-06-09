import networkx as nx
from utils.b_chromatic_utils import compute_remaining_colors, pick_avilable_color


def is_pivoted(T, dense_vertices, m):
    if len(dense_vertices) > m: return False
    H = nx.induced_subgraph(T, dense_vertices)
    CC = list(nx.connected_components(H))
    if len(CC) < 2: return False
    Ts = []
    for cc in CC:
        #print("Connected component: {}".format(cc))
        #if Ti is not an isolated vertex and Ti is not K2
        #print(" Checking if Ti is not an isolated vertex or K2")
        if len(cc) != 1 and len(cc)!=2:
            c = 0
            #print(" Checking if Ti does not contain an unique vertex with degree greater than 1")
            for v in cc:
                if H.degree[v] > 1:
                    #print("  {}".format(H.degree[v]))
                    c += 1
            if c > 1: return False
        Ts.append(cc)
    #print(len(Ts))
    #If we had a star we will just pick its center because is the only that can be connected to the pivot
    if len(Ts[0]) > 2:
        T1 = [w for w in Ts[0] if H.degree[w] > 1]
        T2 = Ts[1]
    if len(Ts[1]) > 2:
        T1 = [w for w in Ts[1] if H.degree[w] > 1]
        T2 = Ts[0]
    else:
        T1 = Ts[0]
        T2 = Ts[1]
        
    #constant foor loop becasue the connected components are at most K2 or the center of the star
    #we will have at most 4 inspections for m-1 vertices so the complexity is O(m)
    pivot = None
    for u in T1:
        for v in T2:
            for w in T.adj[v]:
                if w in T.adj[u]:
                    pivot = w
                    break
    
    #print(pivot)
    if pivot is None: return False
    #after getting the pivot we should verify properties 2 and 3 from definition of the pivot
    
    dist_vertices = {v:-1 for v in T.nodes()}
    dense_mask = {v:False for v in T.nodes()}
    for v in dense_vertices: dense_mask[v]=True
    
    #property 2: each dense vertex is adjacent either to v or to a dense vertex adjancent to v
    #run a BFS to verify if v is found at some point
    #property 3: Any dense vertex adjacent to v and to another dense vertex has degree m-1
    #during the BFS we are going to ask each vertex 2-distance from the pivot if its father has degree m-1
    q = [pivot]
    dist_vertices[pivot] = 0
    while len(q)!=0:
        v=q.pop(0)
        for u in T.adj[v]:
            if dist_vertices[u]==-1:
                dist_vertices[u] = dist_vertices[v]+1
                q.append(u)
                if dense_mask[u]:
                    if dist_vertices[u]>2:
                        return False
                    if dist_vertices[u]==2 and T.degree[v] != m-1:
                        return False
                
    return pivot

def get_candidates_v1_v2(T, V1, v, v1, v2):
    candidates = {u:False for u in V1}
    dist_v = {u:-1 for u in T.nodes()}
    dense = {u:False for u in T.nodes}
    for u in V1: dense[u]=True
    dist_v[v] = 0
    q = [v]
    while len(q) != 0:
        u = q.pop(0)
        for w in T.adj[u]:
            if dist_v[w] == -1 and dense[w]:
                q.append(w)
                dist_v[w] = dist_v[u] + 1
                if dist_v[w] == 2: candidates[u]=True
                
    #print(candidates)
                
    if not candidates[v1]:
        for i, w in enumerate(V1[2:]):
            if candidates[w]:
                V1[0] = w
                V1[i+2] = v1
                
    if not candidates[v2]:
        for i, w in enumerate(V1[2:]):
            if candidates[w]:
                V1[1] = w
                V1[i+2] = v2
    
    return V1


def color_pivoted_tree(T, V1, v, m, step=None):
    colours = {u:None for u in T.nodes()}
    dense = {u:False for u in T.nodes()}
    for u in V1: dense[u]=True
    V1 = get_candidates_v1_v2(T, V1, v, V1[0], V1[1])
    
    #we search vr adjacent to v1 s.t. vr is dense
    vr = None
    v1 = V1[0]
    for w in T.adj[v1]:
        if dense[w]:
            vr = w
            break

    #STEP 1: we now color v2...vm
    for i, u in enumerate(V1):
        if i > 1:
            colours[u] = i
    
    if step == 1: return colours


    #STEP 2        
    #and to v1 the color 2
    colours[v1] = 1
    colours[V1[1]] = 1
    #we assign to the pivot the color of vr
    colours[v] = colours[vr]

    if step == 2: return colours
    
    #STEP 3
    available_colors = set([colour for colour in colours.values() if colour is not None])
    dense_remaining_colors = compute_remaining_colors(T, V1, colours, available_colors)
    for u in V1:
        for v in T.adj[u]:
            if colours[v] is None and len(dense_remaining_colors[u]) != 0:
                colours[v] = dense_remaining_colors[u].pop()
    if step == 3: return colours
    #STEP 4
    non_colored_nodes = [v for v in T.nodes() if colours[v] is None]
    non_colored_nodes_available_colors = compute_remaining_colors(T, non_colored_nodes, colours, available_colors)
    for v in non_colored_nodes:
        colours[v] = pick_avilable_color(T, v, colours, available_colors)
    return colours
