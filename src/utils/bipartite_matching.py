import networkx as nx
import random, copy
from utils.b_chromatic_utils import *

def getNXgraph(G):
    Gnx = nx.Graph()
    for u in G.keys(): 
        for v in G[u]: Gnx.add_edge(u,v)   
    return Gnx

def checkHIsBipartite(G):
    Gnx = getNXgraph(G)  
    return nx.is_bipartite(Gnx)

def getH(G, u, f, C, fixed=None):
    """G: Graph to be recoloured
        u: b-chromatic vertex to which we want to add colour d
        f: colour assignation function
        C: colour set we want to assign to the neighbourhood of u
        fixed: Fixed vertices are not allow to change their colour
    """
    if fixed is None:
        fixed = {u:False for u in G}
    A = ["c{}".format(c) for c in C]
    B = ["v{}".format(v) for v in G[u]]
    H = {}
    for b in B: H[b] = []
    #Then we add an edge (x,y) to H iff colour x is available for vertex y (colour x is not in the neighbourhood of y)
    for c in C:
#        print(c)
        W = [] #set of vertices for which colour c is available
        for v in G[u]:
            c_not_CNv = True
            if f[v] == c: W.append("v{}".format(v))
            else: 
                for w in G[v]:
                    if w != u and f[w] == c: 
#                        print("vertex {} has colour {} on its neighbourhood".format(v, c))
                        c_not_CNv = False
                        break
                if c_not_CNv and not fixed[v]:#If c is not in f(N(v)) and v is not fixed
                    W.append("v{}".format(v))
        if len(W) > 0: H["c{}".format(c)] = W
    #A = {c: len(H[c]) for c in A if u in H}
    #A = sorted(A.items(), key=lambda x:x[1])
    #A = list(zip(*A))[0]
    #print(H)
    A = [c for c in A if c in H]
    assert checkHIsBipartite(H)
    return H, A, B

def shortest_path_s_t(G, s, t):
    dist = {u:-1 for u in G.keys()}
    dist[s] = 0
    Q = []
    parent = {u:None for u in G.keys()}
    Q.append(s)
    P = None
    while len(Q) > 0:
        u = Q.pop(0)
        for v in G[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                Q.append(v)
            if v == t: break
    if dist[t] != -1: 
        P = (s, parent[parent[t]], parent[t], t)
    return P


def augmenting_path(G, M, s, t):
    for path in M:
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            if v in G[u]: G[u].remove(v)
            if u not in G[v]: G[v].append(u)
    P = shortest_path_s_t(G, s, t)
    #print("Path found: {}".format(P))
    return [P]
    

def bipartite_matching(G, A, B):
    s = "s"
    t = "t"
    Gc = copy.deepcopy(G)
    Gc[s] = [u for u in A]
    Gc[t] = []
    for u in B: Gc[u].append(t)
    M = []#set of paths (edges)
    while(True):
        P = augmenting_path(Gc, M, s, t)
        if P == [None]: break
        Mp = []
        for p in P: 
            if p not in M: Mp.append(p)
        for p in M: 
            if p not in P: Mp.append(p)
        M = Mp
        #print("Matching: {}".format(M))
    return M

def getColourMaps(M):
    """Given a matching M returns the corresponding colour assignations"""
    colour_assig = {}
    for path in M:
        c = int(path[1][1:])
        u = int(path[2][1:])
        colour_assig[u] = c
    return colour_assig

if __name__ == "__main__":
    ### TEST ###
    n = 10
    G = nx.random_graphs.gnp_random_graph(n, 0.50)
    f = {u:None for j, u in enumerate(G.nodes())}

    #Arbitrary number of colours for testing
    k = n//2
    C = [i for i in range(1,k+1)]

    d = random.choice(C)#we pick a random colour that is not used
    u = G[0]
    H, A, B = getH(G, u, f, d)
    Hnx = getNXgraph(H)

    M = bipartite_matching(H, A, B)

    for p in M:
        print(p)