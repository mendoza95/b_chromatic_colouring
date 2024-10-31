import sys, os, random, copy, time
import networkx as nx
import pandas as pd
from datetime import datetime
from b_chromatic_utils import is_proper, get_b_chromatic_vertices, are_b_chromatic
from bipartite_matching import getH, getNXgraph

def get_b_chromaticness(G, f, C):
    """ Get the b-chromaticness for every colour, i.e., how close to have a b-chromatic vertex a colour is
        G: Graph,
        f: colour assignation function
        C: Set of colours
    """
    b_chromatic = {c:False for c in C}
    BC = {c:-1 for c in C}
    for c in C:
        for u in G:
            if f[u] == c:
                bc = len({f[v] for v in G[u] if f[v] is not None})
                if bc > BC[c]:
                    BC[c] = bc
        if BC[c] == len(C)-1: b_chromatic[c] = True
    return BC, b_chromatic

def is_there_not_realized_colour(C, BC):
    """ Returns the colour with minimum b-chromacity; False otherwise
        C: Set of colours
        BC: b-chrmacity of colours in C
    """
    #print(BC)
    min_chromaticness_c = min(BC.items(), key=lambda x:x[1])[0]
    if BC[min_chromaticness_c] == len(C)-1: return False
    else: return min_chromaticness_c

def get_neighbourhood_presency_matrix(G, f):
    NP = {u:{c:False for c in set(f.values())} for u in G}
    for u in G:
        for v in G[u]:
            NP[u][f[v]] = True
    return NP

def partition_redistribution1(G, f):
    """ Partition redistribution heuristic that finds a b-chromatic colouring of G.
        G: Graph, 
        f: initial colour assignment function
    """
    C = set(list(f.values()))
    #print("Initial number of colours: {}".format(len(C)))
    fp = copy.deepcopy(f)
    Cp = C.copy()
    for c in C:
        Vc = [u for u in G.nodes() if fp[u] == c]
        Vcp = set(Vc.copy())
        random.shuffle(Vc)
        vertices_to_recolour = {}
        for v in Vc:
            cNv = {fp[u] for u in G.adj[v]}
            if len(cNv) < len(Cp)-1:
                #we compute the colours in N(v)
                colours_diff = Cp.difference(cNv.union(set([c])))
                assert colours_diff != set()
                #we pick a random colour
                cp = random.choice(list(colours_diff))
                vertices_to_recolour[v] = cp
                Vcp.remove(v)
        if Vcp == set():
            for v, cp in vertices_to_recolour.items():
                fp[v] = cp
            Cp.remove(c)
    #print("New number of colours: {}".format(len(set(list(fp.values())))))
    return fp

def partition_redistribution2(G, f):
    C = set(list(f.values()))
    fp = copy.deepcopy(f)
    Cp = C.copy()
    c_class_size = {c:1 for c in C}#sizes of colour classes for every colour
    #V = {u:G.degree(u) for u in G}
    #V = sorted(V.items(), key=lambda x:x[1])
    #V = list(zip(*V))[0]
    V = list(G.nodes())
    BC, _ = get_b_chromaticness(G, f, Cp)
    random.shuffle(V)
    while(True):
        c = is_there_not_realized_colour(Cp, BC)
        if c is False: break
        #print("Deleting colour: {}".format(c))
        Vc = {u for u in V if fp[u] == c}
        Vcp = Vc.copy()
        vertices_to_recolour = {}
        for v in Vc:
            cNv = {fp[u] for u in G.adj[v]}
            if len(cNv) < len(Cp)-1:
                #we compute the colour class sizes for the colours in (N(v)\cup \{v}
                CNv = {cp: c_class_size[cp] for cp in Cp.difference(cNv.union(set([c])))}
                assert CNv != set()
                #we compute the minimum colour size class
                min_colour_size_class = min(CNv.items(), key=lambda x:x[1])[1]
                #we compute a list of colours with minimum colour size class
                min_items = [x for x, y in CNv.items() if y == min_colour_size_class]
                #we pick a random colour with minimum colour size class
                cp = random.choice(min_items)
                vertices_to_recolour[v] = cp
                Vcp.remove(v)
        if Vcp == set():
            for v, cp in vertices_to_recolour.items():
                fp[v] = cp
                #we update the colour class size of colour cp
                c_class_size[cp] += 1
            Cp.remove(c)
            BC, _ = get_b_chromaticness(G, fp, Cp)
    #print("New set of colours: {}".format(set(list(fp.values()))))
    return fp

def partition_redistribution3(G, f):
    C = set(list(f.values()))
    fp = copy.deepcopy(f)
    Cp = copy.deepcopy(C)
    BC, _ = get_b_chromaticness(G, fp, C)
    NP = get_neighbourhood_presency_matrix(G, fp)
    V = list(G.nodes())
    random.shuffle(V)
    #for c in C:
    while(True):
        c = is_there_not_realized_colour(Cp, BC)
        if c == False: break
        #print("Deleting colour {}".format(c))
        #print(BC)
        Vc = {u for u in V if fp[u] == c}
        Vcp = Vc.copy()
        vertices_to_recolour = {}
        for v in Vc:
            cNv = {fp[u] for u in G.adj[v]}
            if len(cNv) < len(Cp)-1:
                #here is where we distinguish between candidates and helpers
                if G.degree(v) >= len(Cp)-1:#v is a candidate
                    #we'll assign v the colour that's more close to become b-chromatic
                    BCp = {cpp:BC[cpp] for cpp in Cp.difference(cNv.union({c}))}
                    #we compute the maximum b-chromacity over all colours
                    max_bc = max(BCp.items(), key=lambda x:x[1])[1]
                    #we make a list of colours with maximum b-chromacity
                    max_items = [x for x, y in BCp.items() if y == max_bc]
                    #and we pick a random colour with maximum b-chromacity
                    cp = random.choice(max_items)

                #v is a helper; we'll assign to v a colour c not in N(v) that appear less times in N(N(v))
                else:
                    #we compute the set of colours c not in N(v) that appear less times in N(N(v))
                    NPc = {cpp:0 for cpp in Cp.difference(cNv.union({c}))}
                    for cpp in NPc:
                        for u in G[v]:
                            #we use the presence matrix to compute the number of times colour c appears in N(u) for u in N(v)
                            if NP[u][cpp]: NPc[cpp] += 1
                    #we compute the minimum colour precensy on the N(N(v))
                    min_c_presency  = min(NPc.items(), key=lambda x:x[1])[1]
                    #we make a list of colours with minimum presency
                    min_items = [x for x, y in NPc.items() if y == min_c_presency]
                    #we pick a random colour with minimum presency
                    cp = random.choice(min_items)
                vertices_to_recolour[v] = cp
                Vcp.remove(v)
        if Vcp == set():
            for v, cp in vertices_to_recolour.items():
                #print("  Changing the colour of {} from {} to {}".format(v, c, cp))
                fp[v] = cp
            Cp.remove(c)
            BC, _ = get_b_chromaticness(G, fp, Cp)
            NP = get_neighbourhood_presency_matrix(G, fp)
    return fp

def initial_colouring(G):
    """ Return an initial colouring of G with DELTA(G) colours.
        The colour is found by applying the matching estrategy to colour vertices
    """
    DELTA = max([G.degree(u) for u in G])+1
    C = set([i for i in range(1, DELTA+1)])
    DV = {u:G.degree(u) for u in G}
    V = sorted(DV.items(), key=lambda x:x[1], reverse=True)
    V = list(zip(*V))[0]
    #print(V, G.degree(V[0]))
    f = {u:None for u in G}
    F = {u:False for u in G}
    for u in V:
        cNu = [f[v] for v in G[u] if f[v] is not None]
        #print(cNu)
        if f[u] is None:
            c = min(C.difference(set(cNu)))
            #print("f[{}] <- {}".format(u, c))
            f[u] = c
            F[u] = True
        H, X, Y = getH(G, u, f, C.difference(set(cNu).union({f[u]})), F)
        Hn = getNXgraph(H)
        M = nx.bipartite.maximum_matching(Hn, X)
        k = len(M)/2
        #print(M)
        while(len(M) > k):
            w, cp = M.popitem()
            cp = int(cp[1:])
            w = int(w[1:])
        #    print(" f[{}] <- {}".format(w, cp))
            f[w] = cp
            F[w] = True
    return f

def test_partition_redistribution(G, pr_function):
    #Initial colouring
    f = {u:j+1 for j, u in enumerate(G.nodes())}
    start = time.time()
    fp = pr_function(G, f)
    end = time.time()

    #re indexing colours
    g = {u:i+1 for i, u in enumerate(set(fp.values()))}
    for u in G:
        f[u] = g[fp[u]]

    Vp = get_b_chromatic_vertices(G, f)
    assert is_proper(G, f)
    assert are_b_chromatic(G, f, Vp)

    k = len(set(f.values()))

    return k, end-start, f

def run_test(G, f, pr_function, N):
    """ Runs a partition redistribution algorithm N times and return the best result and computation time
        G: Graph,
        f: Colour assignment function,
        pr_function: Partition redistribution function
        N: Number of iterations
    """
    best_phiG = 0
    start = time.time()
    for _ in range(N):
        fp = pr_function(G, f)
        Cp = set(fp.values())
        Vp = get_b_chromatic_vertices(G, fp)
        assert is_proper(G, fp)
        assert are_b_chromatic(G, fp, Vp)
        if len(Cp) > best_phiG: best_phiG = len(Cp)
    end = time.time()
    return fp, best_phiG, end-start

if __name__ == "__main__":
    k = int(sys.argv[1])
    now = datetime.now()
    folder = "partition_redistribution_results/{}"
    if os.path.exists(folder[:-3]) is False:
        os.mkdir(folder[:-3])
    filename = "{}_{}_{}_{}_{}_stats.csv".format(now.day, now.month, now.year, now.hour, now.minute)
    run_test(k, folder.format(filename))