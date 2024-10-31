import copy, time
import networkx as nx
from b_chromatic_utils import get_b_chromatic_vertices, is_proper, are_b_chromatic
from bipartite_matching import getNXgraph, getH

def preprocessing(G, f):
    #Set of colours and number of colours
    C = set(f.values())
    k = len(C)

    #colours sorted by number of vertices
    CDu = {c:0 for c in C}
    for u in G: CDu[f[u]] += 1

    CDu = sorted(CDu.items(), key=lambda x:x[1])
    C = list(list(zip(*CDu))[0])

    #b-chromatic vertices
    Vb = get_b_chromatic_vertices(G, f)
    B = {u:False for u in G}
    for u in Vb: B[u]=True
    
    #set of b-chromatic vertices with degree at least k
    BD = {u:False for u in G}
    for u in G:
        if G.degree(u)>= k and B[u]: BD[u] = True

    #set of vertices sorted by degreee
    V = {u:G.degree(u) for u in G}
    V = sorted(V.items(), key=lambda x:x[1])
    V = list(zip(*V))[0]
    #print(V)

    #Set of colours for which there exists a vertex v with degree at least k
    Cp = set()
    for u in G:
        if G.degree(u): Cp.add(f[u])

    #Set of colours for which there is no vertex with degree at least k
    Cpp = [c for c in C if c not in Cp]

    #Set of fixed vertices (A fixed vertex is a vertex that should not change its colour)
    F = {u:False for u in G}

    return V, BD, C, Cp, Cpp, F

def add_colour_k_plus_1(G, C, Cp, Cpp, V, f, BD, F, b_chromatic, bipartites_graphs):
    """ Adds colour |C|+1 to the neighbours of the b-chromatic vertices
        G: Graph
        C: Current set of colours
        Cp: Set of colours for which there exists b-chromatic vertices with degree at least |C|
        Cpp: Complement of Cp with respect to C
        V: Set of vertices sorted by degree in non descendant order
        f: colour assingment function
        BD: Set of b-chromatic vertices with degree at least |C|
        F: Set of fixed vertices
    """
    k = len(C)
    for c in Cp:
        #print("Looking for a b-chromatic vertex of colour {}".format(c))
        success = False
        for u in V:
            #print("Inspecting vertex {}".format(u))
            if BD[u] and not b_chromatic[u] and f[u] == c:
                NCu = [cp for cp in C+[k+1] if cp !=  f[u]]
                #NCu = C.difference({f[u]}).union({k+1})
                H, X, Y = getH(G, u, f, NCu, F)
                Hn = getNXgraph(H)
                M = nx.bipartite.maximum_matching(Hn, X)
                if len(M)/2 == k:
                    bipartites_graphs[c]["H"] = H
                    bipartites_graphs[c]["X"] = X
                    bipartites_graphs[c]["Y"] = Y
                    bipartites_graphs[c]["M"] = M
                    #print("Colour of {}: {} \n\t Assigning to N({}) colours: {}".format(u, f[u], u, NCu))
                    #print("\tb-chromatic vertex for colour {} found: {}".format(c, u))
                    for _ in range(k): 
                        w, cp = M.popitem()
                        cp = int(cp[1:])
                        w = int(w[1:])
                        f[w] = cp
                        F[w] = True
                        #print("\t f[{}] = {}".format(w, cp))
                    b_chromatic[u] = True
                    success = True
                    break
        if(not success): Cpp += [c]

    return f, Cpp, F, b_chromatic, bipartites_graphs

def find_b_chromatic_k_plus_1(G, V, C, Cpp, fc, F, b_chromatic, bipartites_graphs):
    """ Search for b-chromatic vertices for |C|+1 colours
        G: Graph
        V: Set of vertices sorted by degree in non descending order
        C: Current set of colours
        Vp: Set of candidates to become b-chromatic vertices
        Cpp: Set of colours for which there have not been found b-chromatic vertices by extending the colouring f
        fc: Colouring function
        F: Set of fixed vertices
        b_chromatic: Binary vector of b-chromatic vertices in the extended colouring
        bipartite_graphs: Store metadate about the assignation of colours
    """
    k = len(C)
    Vp = [u for u in V if G.degree(u) >= k and not b_chromatic[u]]
    for c in Cpp:
        #print("Looking for a b-chromatic vertex of colour {}".format(c))
        if len(Vp) == 0: break
        success = False
        for u in Vp:
            d = -1
            cNu = {cp:False for cp in C+[k+1]}
            for v in G[u]: cNu[fc[v]] = True
            if (fc[u] == c or (fc[u]!= c and not cNu[c])) and not F[u]:
                if fc[u] != c:
                    #print("We change the colour of {} from {} to {}".format(u, fc[u], c))
                    d = fc[u]
                    fc[u] = c
                NCu = [cp for cp in C+[k+1] if cp !=  fc[u]]
                #NCu = C.union({k+1}).difference({fc[u]})
                H, X, Y = getH(G, u, fc, NCu, F)
                Hn = getNXgraph(H)
                M = nx.bipartite.maximum_matching(Hn, X)
                if len(M)/2 == k:
                    bipartites_graphs[fc[u]]["H"] = H
                    bipartites_graphs[fc[u]]["X"] = X
                    bipartites_graphs[fc[u]]["Y"] = Y
                    bipartites_graphs[fc[u]]["M"] = M
                    #print("Colour of {}: {} \n\t Assigning to N({}) colours: {}".format(u, fc[u], u, NCu))
                    #print("\tb-chromatic vertex for colour {} found: {}".format(fc[u], u))
                    for _ in range(k): 
                        w, cp = M.popitem()
                        cp = int(cp[1:])
                        w = int(w[1:])
                        #print("\t f[{}] = {}".format(w, cp))
                        fc[w] = cp
                        F[w] = True
                    b_chromatic[u] = True
                    success = True
                    break
                elif d != -1: fc[u] = d
        if not success:
            #print("Not possible to find a b-chromatic colouring for colour {}".format(c))
            return success, fc, b_chromatic, F, bipartites_graphs
        else:
            Vp.remove(u)
    return True, fc, b_chromatic, F, bipartites_graphs

def extend_b_chromatic_colouring_by_one1(G, f):
    """Extend the colouring f of G by 1 maintaining the b-chromatic property
        G: Graph
    """
    print("PREPROCESSING", end="\r")
    V, BD, C, Cp, Cpp, F = preprocessing(G, f)

    #AUXILIAR DICTIONARY FOR BIPARTITE GRAPH H
    bipartites_graphs = {c:{"H":None, "X":None, "Y":None, "M":None} for c in C+[len(C)+1]}

    fc = copy.deepcopy(f)
    b_chromatic = {u:False for u in G}

    print("STAGE (1)", end="\r")
    fc, Cpp, F, b_chromatic, bipartites_graphs = add_colour_k_plus_1(G, C, Cp, Cpp, V, fc, BD, F,
                                                                     b_chromatic,
                                                                     bipartites_graphs)

    print("STAGE (2)", end="\r")
    Cpp = Cpp+[len(C)+1]
    success, fc, b_chromatic, F, bipartites_graphs = find_b_chromatic_k_plus_1( G, V, C, Cpp, fc, F, 
                                                                                b_chromatic, 
                                                                                bipartites_graphs)
    Bp = [u for u in G if b_chromatic[u]]

    if not success:
#        print("NO EXTENSION!")
        return None
    else:
        if not is_proper(G, fc): 
#            print("NO EXTENSION!")
            return None
        if not are_b_chromatic(G, fc, Bp): 
#            print("NO EXTENSION!")
            return None
        #print("COLOURING EXTENDED!")
        return fc
    
def extend_b_chromatic_colouring_by_one2(G, f):
    """Extend the colouring f of G by 1 maintaining the b-chromatic property
        G: Graph
    """
    print("PREPROCESSING", end="\r")
    V, BD, C, Cp, _, F = preprocessing(G, f)

    #AUXILIAR DICTIONARY FOR BIPARTITE GRAPH H
    bipartites_graphs = {c:{"H":None, "X":None, "Y":None, "M":None} for c in C+[len(C)+1]}

    fc = copy.deepcopy(f)
    b_chromatic = {u:False for u in G}

    print("STAGE (1)", end="\r")
    Cpp = [len(C)+1]
    success, fc, b_chromatic, F, bipartites_graphs = find_b_chromatic_k_plus_1( G, V, C, Cpp, fc, F, 
                                                                                b_chromatic, 
                                                                                bipartites_graphs)

    if not success:
        #print("NO EXTENSION!")
        return
    
    print("STAGE (2)", end="\r")
    Cpp.pop()
    fc, Cpp, F, b_chromatic, bipartites_graphs = add_colour_k_plus_1(G, C, Cp, Cpp, V, fc, BD, F,
                                                                     b_chromatic,
                                                                     bipartites_graphs)
    if len(Cpp) > 0:
        #print("STILL {} B-CHROMATIC VERTICES TO FOUND - NO EXTENSION!".format(len(Cpp)))
        print("STAGE (3)", end="\r")
        success, fc, b_chromatic, F, bipartites_graphs = find_b_chromatic_k_plus_1( G, V, C, Cpp, fc, F, 
                                                                                    b_chromatic, 
                                                                                    bipartites_graphs)
        
    Bp = [u for u in G if b_chromatic[u]]

    if not success:
        #print("NO EXTENSION!")
        return None
    else:
        if not is_proper(G, fc): 
        #    print("NO PROPER COLOURING - NO EXTENSION!")
            return None
        if not are_b_chromatic(G, fc, Bp): 
        #    print("NOT ALL B-CHROMATIC - NO EXTENSION!")
            return None
        #print("COLOURING EXTENDED!")
        return fc
    
def extend_colouring(G, f, f_extension):
    """Extends a colouring as many times as possible
        G: Graph,
        f: colouring function
        f_extension: function to extend the colouring
    """
    #print("Extending the colouring as many times as possible")
    fc = copy.deepcopy(f)
    total_time = 0
    b_chr_ext = None
    while(True):
        start = time.time()
        fc = f_extension(G, fc)
        end = time.time()
        if fc is None: 
            break
        else:
            if b_chr_ext is None:
                b_chr_ext = len(set(fc.values()))
            else: b_chr_ext += 1
        total_time += end-start
    #print("Colouring extended {} times".format(b_chr_ext))
    if b_chr_ext is None: return 0, total_time
    else: return b_chr_ext, total_time