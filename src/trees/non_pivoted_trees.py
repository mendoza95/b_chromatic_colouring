import networkx as nx
from utils.b_chromatic_utils import compute_remaining_colors, pick_avilable_color
from pivoted_trees import get_candidates_v1_v2

def does_encircle(T, V2, m):
    #T is the tree
    #V_2 is the subset of dense vertices
    #Return the encircle vertex if exists; None otherwise
    assert len(V2) == m
    
    H = nx.subgraph(T, V2)
    CC = nx.connected_components(H)
    Ts = [cc for cc in CC]
    
    if len(Ts)<2: return False
    
    T1 = Ts[0]
    T2 = Ts[1]
    
    #If we had a star we will just pick its center because is the only that can be connected to the pivot
    if len(T1) > 2:
        T1 = [w for w in T1 if H.degree[w] > 1]
    if len(T2) > 2:
        T2 = [w for w in T2 if H.degree[w] > 1]
        
    #We pick a candidate for the encircled vertex
    #constant foor loop becasue the connected components are at most K2 or the center of the star
    #we will have at most 4 inspections for m-1 vertices so the complexity is O(m)
    encircle_v = None
    for u in T1:
        for v in T2:
            for w in T.adj[v]:
                if w in T.adj[u]:
                    encircle_v = w
                    break
                    
    if encircle_v is None: return False
    
    #We check if the candidate is encircle
    dist_vertices = {v:-1 for v in T.nodes()}
    dense_mask = {v:0 for v in T.nodes()}
    for v in V2: dense_mask[v]=1
        
    #property 2: each dense vertex is adjacent either to v or to a dense vertex adjancent to v
    #We run a 2 level BFS on v to verify the dense vertices found
    #property 3: Any dense vertex adjacent to v and to another dense vertex has degree m-1
    #during the BFS we are going to ask each 2-distance vertex from the pivot if its father has degree m-1
    q = [encircle_v]
    dist_vertices[encircle_v] = 0
    while len(q)!=0:
        v=q.pop(0)
        for u in T.adj[v]:
            if dist_vertices[u]==-1:
                dist_vertices[u] = dist_vertices[v]+1
                q.append(u)
                if dense_mask[u]==1:
                    if dist_vertices[u]>2:
                        return False
                    if dist_vertices[u]==2 and T.degree[v] != m-1:
                        return False
    
    return encircle_v

def find_m_largets_dense(T, V, m):
    m_largest_dense = []

    #for v in V:
    #    if T.degree(v) >= m: m_largest_dense.append(v)
    #for v in V:
    #    if T.degree(v) >= m-1: m_largest_dense.append(v)
    vertices_degree = {v:T.degree(v) for v in V}
    for _ in range(m):
        key = max(vertices_degree, key=lambda k:vertices_degree[k])
        m_largest_dense.append(key)
        vertices_degree.pop(key)
    return m_largest_dense

def good_set(T, V1, m):
    #T --> Tree
    #V --> Set of dense vertices
    #m --> m_degree of T
    #We choose a subset of V with size m; We will choose the first m vertices by default
    V2 = find_m_largets_dense(T, V1, m)
    V1_mask = {u:False for u in T.nodes()}
    V2_mask = {u:False for u in T.nodes()}
    for u in V1: V1_mask[u]=True
    for u in V2: V2_mask[u]=True
    
    v = does_encircle(T, V2, m)
    #print(v)
    if v is False: return V2
    V2 = get_candidates_v1_v2(T, V2, v, V2[0], V2[1])
    #print(V2)
    #if v is a dense vertex
    if V1_mask[v]:
        #exchange v2 with v: delete second vertex from V2 and add v to it
        V2.pop(1)
        V2.append(v)
    #if v is not dense
    else:
        #exchange v1 with u
        for u in T.nodes():
            if V1_mask[u] and not V2_mask[u]:
                V2.pop(0)
                V2.append(u)
                break
    return V2

def finding_good_set_plus_3path_vertices(T, W):
    #Return a forest which is a induced subgraph on the good vertices + vertices that lies in a path of lenght at most 3 between two good vertices
    W_prime = []
    good_mask = {u:False for u in T.nodes()}
    for u in W: good_mask[u]=True
    for u in W:
        W_prime.append(u)
        for v in T.adj[u]:
            W_prime.append(v)
    T_prime = nx.induced_subgraph(T, W_prime).copy()
    #return W_prime,T_prime
    nodes_del = []
    for u in T_prime.nodes():
        if T_prime.degree(u) == 1 and not good_mask[u]:
            nodes_del.append(u)
    
    T_prime.remove_nodes_from(nodes_del)
        
    return T_prime
        
def is_there_2len_path(T, u, dist, good_mask, vis):
    vis[u] = True
    v = None
    if dist == 2:
        if good_mask[u]: return u
        else: return v
    elif dist == 1:
        if not good_mask[u]:
            for w in T.adj[u]:
                if vis[w] is False:
                    v = is_there_2len_path(T, w, dist+1, good_mask, vis)
                    if v is not None: break
    elif dist == 0:
        for w in T.adj[u]:
            if vis[w] is False:
                v = is_there_2len_path(T, w, dist+1, good_mask, vis)
                if v is not None: break
    return v

def is_there_anylen_path(T, u, dist, good_mask, vis):
    vis[u] = True
    v = None
    if dist <= 2:
        if good_mask[u]: return u
        for w in T.adj[u]:
            if vis[w] is False:
                v = is_there_anylen_path(T, w, dist+1, good_mask, vis)
                if v is not None: break
    return v    


def color_non_pivoted_tree(T, W, good_set_mask, m, step=None):
    colors = {u:None for u in T.nodes()}
    #STEP 1
    for i,w in enumerate(W): colors[w]=i+1
    C = set([colour for colour in colors.values() if colour is not None])
    if step == 1: return colors
    
    #STEP 2
    T_prime = finding_good_set_plus_3path_vertices(T, W)
    
    cc = nx.connected_components(T_prime)
    Ts = [c for c in cc]
        
    for c in Ts:
        T1 = nx.induced_subgraph(T, c)
        #print("Vertices from the CC: {}".format(c))
        #print("Stage 1")
        for u in T1.nodes:
            S=[]
            if good_set_mask[u]:
                #print(u)
                for v in T1.adj[u]:
                    if colors[v] is None:
                        S.append(v)
                #print(u,S)
                vis = {u:False for u in T1.nodes()}
                vis[u] = True
                vertices_colors = []
                #after getting S, we should visist process S based on its size
                if len(S) == 1:
                    w = S[0]
                    v = is_there_2len_path(T1, w, 0, good_set_mask, vis)
                    if v is not None:
                        colors[w] = colors[v]
                        #print(" {}->{}".format(w, colors[w]))
                else:
                    for w in S:
                        v = is_there_anylen_path(T1, w, 0, good_set_mask, vis)
                        if v is not None:
                            vertices_colors.append([w,v])
                    n = len(vertices_colors)
                    #print(" Vertices color {}".format(vertices_colors))
                    for i, (w, _) in enumerate(vertices_colors):
                        colors[w] = colors[vertices_colors[(i+1)%n][1]]
                        #print(" {}->{}".format(w, colors[w]))
                    
        
        #print("Stage 2")
        for u in T1.nodes():
            if colors[u] is None:
                #print(u)
                available_colors_u = C.copy()
                for v in T1.adj[u]:
                    available_colors_u.discard(colors[v])
                    if T.degree(v) == m-1:
                        for w in T1.adj[v]:
                            if w != u: available_colors_u.discard(colors[w])
                colors[u] = available_colors_u.pop()
                #print(" {}->{}".format(u, colors[u]))
                        
    #print("Stage 3 {}".format(available_colors))
    #available_colors = set([i for i in range(m)])
    if step == 2: return colors

    #STEP 3
    good_set_remaining_colors = compute_remaining_colors(T, W, colors, C)
    for w in W:
        #print(good_set_remaining_colors[w])
        for v in T.adj[w]:
            if colors[v] is None:
                if len(good_set_remaining_colors[w]) != 0:
                    colors[v] = good_set_remaining_colors[w].pop() 
                    #print(" {}->{}".format(v, colors[v]))

    non_colored_nodes = [v for v in T.nodes() if colors[v] is None]
    #non_colored_available_colors = compute_remaining_colors(T, non_colored_nodes,colors, available_colors)

    if step == 3: return colors

    #STEP 4
    for v in non_colored_nodes:
        colors[v] = pick_avilable_color(T, v, colors, C)
    return colors
