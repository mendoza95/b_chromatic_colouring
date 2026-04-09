import os, sys, random, copy
import networkx as nx
import itertools
from utils.b_chromatic_utils import is_proper, are_b_chromatic
from utils.bipartite_matching import getH, getNXgraph

def find_b_chromatic_colouring_bruteforce(G, d):
    """Search for a b-chromatic colouring with d+1 colours on a d-regular graph G trying out all posible combination of colours
    Parameters -> G: d-regular graph, d: integer
    Returns -> c: colour assignment function or False if no such colouring exists
    """
    C = set([i for i in range(1,d+2)])
    S = list(itertools.combinations(G.nodes(),d+1))
    colour_permutations = [list(itertools.permutations(C.difference({i}))) for i in C]
    colour_prod_permutations = list(itertools.product(*colour_permutations))
    for node_sets in S:
        c = {u:None for u in G.nodes()}
        for i, vi in enumerate(node_sets): c[vi] = i+1
        for Xs in colour_prod_permutations:
            for i, vi in enumerate(node_sets):
                for j, uj in enumerate(G.adj[vi]):
                    if uj not in node_sets:
                        c[uj] = Xs[i][j]
            if is_proper(G, c) and are_b_chromatic(G, c, node_sets):
                print("b-chromatic vertices: {}".format(node_sets))
                return c
    return False

#A more efficient implementation rely on the search of a perfect matching between the colours and the available colours in the neighbourhood of x
def find_b_chromatic_assignment(G, Cp, x, c):
    """Search through all possible combination of colours in Cp an assigment that makes x b-chromatic(provided that such assignment exists)
    G: graph,
    Cp: Colours that need to be assign to N(u)
    x: vertex to check, 
    c: colouring function, 
    success: True when a b-chromatic assigment has been found"""
    C = list(itertools.permutations(list(Cp)))
    #T = {y for y in G.adj[x]}
    for colours in C:
        cp = c.copy()
        for i, xi in enumerate(G.adj[x]):
            if cp[xi] is None:
                cp[xi] = colours[i]
        #print(is_proper(G, cp), are_b_chromatic(G, cp, {x}))
        #print(cp)
        if is_proper(G, cp) and are_b_chromatic(G, cp, {x}): return cp, True
    return c, False

def find_b_chromatic_assignment2(G, C, x, c, d):
    cNx = {c[v] for v in G.adj[x]}
    F = {u:False for u in G}
    for u in G[x]:
        if c[u] != None: F[u] = True
    H, X, Y = getH(G, x, c, C.difference(set(cNx).union({c[x]})), F)
    Hn = getNXgraph(H)
    M = nx.bipartite.maximum_matching(Hn, X)
    k = int(len(M)/2)
    if k < len(X): return c, False
    f = copy.deepcopy(c)
    while(len(M) > k):
        w, cp = M.popitem()
        cp = int(cp[1:])
        w = int(w[1:])
        f[w] = cp
    f[x] = d
    if is_proper(G, f) and are_b_chromatic(G, f, {x}): return f, True
    else: return f, False

def does_conditions_i_ii_lemma3_hold(G, Cp, c, x, CSNx, cNx):
    """Checks whether condition I and II of Lemma 3 hold on the uncoloured neighbours of x
    G: Graph
    Cp: set of colours that vertex x need to pick up on its neighbourhood
    c: colour function assigment
    x: candidate vertex to become b-chromatic
    SNx: Subset of neighbours of x with no colour assign to it (|CSNx| >= 2)
    cNx: Set of colour used in N(x)"""
    cCSNx = {xp: {c[xpp] for xpp in G.adj[xp] if xpp != x and c[xpp] is not None}for xp in CSNx}
    assert cCSNx != set()
    I = Cp.difference(cNx)
    for xp in cCSNx.keys(): I = I.intersection(cCSNx[xp])
    if I != set(): return False
    for xp1, xp2 in itertools.combinations(CSNx, 2):
        if (len(cCSNx[xp1].intersection(cCSNx[xp2])) == 2): return False
    return True

def can_u_become_b_chromatic(G, C, u, d, c):
    """Checks whether vertex u can become a b-chromatic vertex of colour d
    G: Graph
    C: Set of colours used in the colouring c
    u: vertex to be checked
    d: colour that vertex u will pick up
    c: colour assignation function
    """
    for up in G.adj[u]:
        if c[up] == d: return False#First we check whether or not d is in the neighbourhood of u
    cNu = {c[up] for up in G.adj[u] if c[up] is not None} #colour set of the neighbour of u
    SNx = {up for up in G.adj[u] if c[up] is not None} #set of coloured neighbours of u
    #print("coloruhood of u: {} and coloured vertices in N(u): {}".format(cNu, SNx))
    if (cNu != set() and SNx != set()) and len(cNu) == len(SNx): return True
    if (cNu != set() and SNx != set()) and len(cNu) < len(SNx): return False
    CSNx = {up for up in G.adj[u] if up not in SNx}
    return does_conditions_i_ii_lemma3_hold(G, C.difference({d}), c, u, CSNx, cNu)

def bfs_dist_4(G, u):
    """Find a vertex path P starting a u and ending at a vertex v such that
      dist(u,v)=4 providing that diam(G) >= 4
    Parameters -> G: graph, u: vertex
    Returns -> P: a path from u to v such that ||P|| = 4 (number of edges)
    """
    dist = {v:-1 for v in G.nodes()}
    #For each i = 1,2,3,4 it stores a pair key:value where key is the vertex at distance i from u 
    # and value is its antecesor in the BFS tree
    distances = {}
    dist[u] = 0
    Q = [u]
    P = []
    while(len(Q) > 0):
        w = Q.pop(0)
        for v in G.adj[w]:
            if(dist[v] == -1):
                dist[v] = dist[w] + 1
                if dist[v] not in distances.keys(): distances[dist[v]] = {v:w}
                else: distances[dist[v]][v] = w
                Q.append(v)
    w = next(iter(distances[4]))
    while(len(P) < 4):
        P.insert(0, w)
        w = distances[5-len(P)][w]
    P.insert(0,u)
    return P


def find_3_b_chromatic_vertices(G, P):
    """Make b-chromatic vertices u, w and v along the path P = <u, u3, w, v3, v>
    Parameters -> G: graph, P: path on G
    Returns -> c: colur assignment function, Q: P+P'+{w'}"""
    c = {u:None for u in G.nodes()}
    u, u3, w, v3, v = P[0], P[1], P[2], P[3], P[4]
    #print(u,w,v)
    u1, u2 = tuple(set(G.adj[u]).difference({u3}))
    v1, v2 = tuple(set(G.adj[v]).difference({v3}))
    #print(u1,u2)
    #print(v1,v2)
    Pp = [u1,u2,v1,v2]
    c[u] = c[v3] = 1
    c[v] = c[u3] = 3 
    c[w] = c[u1] = c[v1] = 2
    c[u2] = c[v2] = 4
    wp = [x for x in G.adj[w] if x != u3 and x != v3][0]
    #print(wp)
    if wp in Pp:
        #print("w' in P'")
        if wp == u1: c[u1], c[u2] = c[u2], c[u1]
        elif wp == v1: c[v1], c[v2] = c[v2], c[v1]
    else:
        #print(wp, u, v)
        S = {x for x in G.adj[wp] if x in Pp}
        #print("S = {}".format(S))
        if len(S) == 0: c[wp] = 4
        elif len(S) == 1:
            
            # Assume S={x}
            x = S.pop()
            if x == u2: c[u1], c[x] = c[x], c[u1]
            if x == v2: c[v1], c[x] = c[x], c[v1]
            c[wp] = 4
        else:
            #Assume S={x,y}
            x, y = S.pop(), S.pop()
            #x in N(u) and y in N(v)
            if x in G.adj[u] and y in G.adj[v]:
                if x == u2: c[u1], c[x] = c[x], c[u1]
                if y == v2: c[v1], c[y] = c[y], c[v1]
            elif x in G.adj[v] and y in G.adj[u]:
                if x == v2: c[v1], c[x] = c[x], c[v1]
                if y == u2: c[u1], c[y] = c[y], c[u1]
            elif x in G.adj[u] and y in G.adj[u]:
                c[v2], c[v3] = c[v3], c[u2]
                c[wp] = c[u]
            elif x in G.adj[v] and y in G.adj[v]:
                c[u2], c[u3] = c[u3], c[v2]
                c[wp] = c[v]
    Q = P+[u1,u2,v1,v2,wp]
    return c, Q

def find_fourth_b_chromatic_vertex(G, x, c, visited, Vp, success, d):
    """ G: Graph, 
        x: vertex to check, 
        c: colouring function, 
        visited: verifies whether a vertex has been visited or not
        S: {u,w,v}, 
        success: True when the fourth b-chromatic vertex has been found"""
    C = {a for a in c.values() if a is not None}
    if not visited[x]:
        visited[x] = True
        cNx = {c[xp] for xp in G.adj[x] if c[xp] is not None}
        if (c[x] == d or (c[x] is None and d not in cNx)):
            if (can_u_become_b_chromatic(G, C, x, d, c)):
                c, success = find_b_chromatic_assignment2(G, C.difference({d}), x, c, d)
                if success:
                    if c[x] is None: c[x] = d
                    Vp.append(x)
                    return c, success
        for xp in G.adj[x]:
            if not visited[xp]:
                c, success = find_fourth_b_chromatic_vertex(G, xp, c, visited, Vp, success, d)
                if success:
                    return c, success
    return c, False
                
def get_b_chromatic_colouring(G, step=None, P=None, Q=None):
    u = random.choice(list(G.nodes()))
    #STEP 1
    P = bfs_dist_4(G, u)
    c, Q = find_3_b_chromatic_vertices(G, P)
    if step == 1: return c

    #STEP 2
    Vp = [P[0], P[2], P[4]]#set of b-chromatic vertices
    visited = {x:False for x in G.nodes()}
    success = False
    while not success:
        x = Q.pop()
        c, success = find_fourth_b_chromatic_vertex(G, x, c, visited, Vp, success, 4)
    assert are_b_chromatic(G, c, Vp)
    assert {c[w] for w in Vp} == {d for d in c.values() if d is not None}
    return c

if __name__ == "__main__":
    pass
