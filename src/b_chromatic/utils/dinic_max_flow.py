import copy

def getGL(G, s):
    levels = {u:-1 for u in G.keys()}
    levels[s] = 0
    Q = [s]
    while len(Q) > 0:
        u = Q.pop(0)
        for v in G[u].keys():
            if (levels[v] == -1) and G[u][v]["c"] > 0:
                levels[v] = levels[u] + 1
                Q.append(v)
    return levels

def dfs_flow(G, levels, u, t, path_found, min_f, P):
    if u == t: return True, min_f
    #print("Visiting vertex {}".format(u))
    for v in G[u].keys():
        #print(" Level of {}: {} and level of {}: {}".format(u, levels[u], v, levels[v]))
        if levels[v] > levels[u] and G[u][v]["c"] > 0:
            #print("({},{}) with remaining capacity: {}".format(u,v,G[u][v]["c"]))
            min_f = min(min_f, G[u][v]["c"])
            path_found, min_f = dfs_flow(G, levels, v, t, path_found, min_f, P)
            if path_found: 
                P.insert(0,v)
                G[u][v]["c"] -= min_f
                G[u][v]["f"] += min_f
                return path_found, min_f
    return path_found, min_f

def find_blocking_flow(G, levels, s, t):
    Ps = []
    Fs = []
    blocking_flow_found = False
    while not blocking_flow_found:
        min_f = 100000
        path_found = False
        P = []
        path_found, min_f = dfs_flow(G, levels, s, t, path_found, min_f, P)
        if len(P) > 0: 
            P.insert(0,s)
            #print("Minimum flow found: {} with path {}".format(min_f, P))
            Ps.append(P)
            Fs.append(min_f)
            if not path_found: blocking_flow_found = True
        else: break
    return blocking_flow_found, Ps, Fs

def getMaxFlow(G, s, t):
    Gr = copy.deepcopy(G)
    paths = []
    flows = []
    while True:
        levels = getGL(Gr, s)
        blocking_flow_found, Ps, Fs = find_blocking_flow(Gr, levels, s, t)
        #if blocking_flow_found:
            #print("Blocking flow found!")
        if len(Ps) > 0:
            paths.extend(Ps)
            flows.extend(Fs)
        else: break
    return paths, flows  

if __name__ == "__main__":
    # First Sample Graph (Residual graph also)
    G1 ={i:{} for i in range(11)}
    G1[0] = {1:{'c':5, 'f':0}, 2:{'c':10, 'f':0}, 3:{'c':15, 'f':0}}
    G1[1] = {4:{'c':10, 'f':0}}
    G1[2] = {1:{'c':15, 'f':0}, 5:{'c':20, 'f':0}}
    G1[3] = {6:{'c':25, 'f':0}}
    G1[4] = {5:{'c':25, 'f':0}, 7:{'c':10, 'f':0}}
    G1[5] = {3:{'c':5, 'f':0}, 8:{'c':30, 'f':0}}
    G1[6] = {8:{'c':20, 'f':0}, 9:{'c':10, 'f':0}}
    G1[7] = {10:{'c':5, 'f':0}}
    G1[8] = {10:{'c':15, 'f':0}}
    G1[9] = {10:{'c':10, 'f':0}}

    paths, flows = getMaxFlow(G1, 0, 10)
    print("Graph 1")
    for i, p in enumerate(paths): print("path {} with minimum flow {}".format(p, flows[i]))
    

    # Second Sample Graph (Residual graph also)
    G2 = {i:{} for i in range(6)}
    G2[0] = {1:{"c":12, "f":0}, 2:{"c":12, "f":0}}
    G2[1] = {2:{"c":4, "f":0}, 3:{"c":6, "f":0}, 4:{"c":10, "f":0}}
    G2[2] = {4:{"c":11, "f":0}}
    G2[3] = {5:{"c":12, "f":0}}
    G2[4] = {3:{"c":8, "f":0}, 5:{"c":12, "f":0}}

    paths, flows = getMaxFlow(G2, 0, 5)
    print("Graph 2")
    for i, p in enumerate(paths): print("path {} with minimum flow {}".format(p, flows[i]))