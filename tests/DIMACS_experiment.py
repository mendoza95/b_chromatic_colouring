import os
import networkx as nx
import pandas as pd
from heuristics.partition_redistribution import *
from heuristics.b_chromatic_colouring_extension import *
from utils.b_chromatic_utils import get_biggest_cc, get_m_degree


name_list = os.listdir("./graphs_datasets/DIMACS/")
graphs_list = [name for name in name_list if name[-3:]=="col" or name[-3:] == "clq"]


#LABED -> QuickBcol
#Fitzer -> HEA_B
#Melo -> MSBCOL
#Montemanni -> IMA
algorithms = [("QuickBcol", "TQ(s)"), 
              ("HEA_B", "TH(s)"), 
              ("MSBCOL", "TM(s)"), 
              ("IMA", "TI(s)")]
other_papers_res = {
    'flat1000_50_0.col':{"HEA_b":{"phiG":139, "t(s)":58427.97},
                         "MSBCOL":{"phiG":153, "t(s)":69.7}},
    'le450_25.d.col':{"HEA_B":{"phiG":45, "t(s)":256.42},
                      "MSBCOL":{"phiG":48, "t(s)":7.2},
                      "IMA":{"phiG":60, "t(s)":1800}},
    'DSJR500.5.col':{"HEA_B":{"phiG":166, "t(s)":15007.00},
                     "MSBCOL":{"phiG":221,"t(s)":">3600"}},
    'R250.5.col':{"HEA_B":{"phiG":92, "t(s)":1836.16}},
    'le450_25.c.col':{"HEA_B":{"phiG":45, "t(s)":571.72},
                      "MSBCOL":{"phiG":48, "t(s)":8.7},
                      "IMA":{"phiG":59, "t(s)":1800}},
    'flat300_28_0.col':{"HEA_B":{"phiG":55, "t(s)":1048.04},
                        "MSBCOL":{"phiG":63, "t(s)":">3600"}},
    'myciel5.col':{"QuickBcol":{"phiG":8, "t(s)":0.7513}},
    'myciel4.col':{"QuickBcol":{"phiG":6, "t(s)":0.5419}},
    'myciel6.col':{"QuickBcol":{"phiG":13, "t(s)":1.8345}},
    'myciel7.col':{"QuickBcol":{"phiG":16, "t(s)":11.3901}},
    'myciel3.col':{"QuickBcol":{"phiG":4, "t(s)":0.4166}},
    'miles1000.col':{"QuickBcol":{"phiG":47, "t(s)":4.2587}},
    'miles250.col':{"QuickBcol":{"phiG":12, "t(s)":0.5527}},
    'queen6.col':{"QuickBcol":{"phiG":11, "t(s)":0.8934}},
    'queen10.col':{"QuickBcol":{"phiG":18, "t(s)":3.5566}},
    'queen5.col':{"QuickBcol":{"phiG":9, "t(s)":0.4326}},
    'DSJC250.5.col':{"HEA_B":{"phiG":50, "t(s)":186.84},
                     "MSBCOL":{"phiG":57, "t(s)":">3600"}},
    'DSJC500.5.col':{"HEA_B":{"phiG":84, "t(s)":4506.87},
                     "MSBCOL":{"phiG":95, "t(s)":">3600"}},
    'DSJC500.1.col':{"HEA_B":{"phiG":25, "t(s)":113.61},
                     "MSBCOL":{"phiG":36, "t(s)":">3600"},
                     "IMA":{"phiG":38, "t(s)":1800}},
    'DSJC125.1.col':{"MSBCOL":{"phiG":17, "t(s)":794.0}},
    'DSJC125.5.col':{"MSBCOL":{"phiG":35, "t(s)":">3600"},
                     "IMA":{"phiG":39, "t(s)":1800}},
    'DSJC125.9.col':{"MSBCOL":{"phiG":68, "t(s)":">3600"}},
    'mulsol.i.1.col':{"MSBCOL":{"phiG":64, "t(s)":11.0}},
    'mulsol.i.2.col':{"MSBCOL":{"phiG":51, "t(s)":1.0}},
    'mulsol.i.3.col':{"MSBCOL":{"phiG":52, "t(s)":1.0}},
    'mulsol.i.4.col':{"MSBCOL":{"phiG":52, "t(s)":"<0.1"}},
    'mulsol.i.5.col':{"MSBCOL":{"phiG":53, "t(s)":1.0}},
    'zeroin.i.1.col':{"MSBCOL":{"phiG":54, "t(s)":1.0}},
    'zeroin.i.2.col':{"MSBCOL":{"phiG":41, "t(s)":1.0}},
    'zeroin.i.3.col':{"MSBCOL":{"phiG":41, "t(s)":2.0}},
    'c-fat200-2.clq':{"MSBCOL":{"phiG":34, "t(s)":"<0.1"}},
    'c-fat200-1.clq':{"MSBCOL":{"phiG":18, "t(s)":"<0.1"}},
    'c-fat200-5.clq':{"MSBCOL":{"phiG":86, "t(s)":"<0.1"}},
    'johnson16-2-4.clq':{"MSBCOL":{"phiG":37, "t(s)":">3600"},
                         "IMA":{"phiG":38, "t(s)":1800}},
    'johnson8-2-4.clq':{"MSBCOL":{"phiG":9, "t(s)":"<0.1"}},
    'hamming6-4.clq':{"MSBCOL":{"phiG":15, "t(s)":">3600"}},
    'MANN_a9.clq':{"MSBCOL":{"phiG":21, "t(s)":"<0.1"}},
    'brock200_2.clq':{"MSBCOL":{"phiG":48, "t(s)":">3600"}},
    'johnson8-4-4.clq':{"MSBCOL":{"phiG":28, "t(s)":">3600"}},
    'hamming6-2.clq':{"MSBCOL":{"phiG":35, "t(s)":3.0}},
    'keller4.clq':{"MSBCOL":{"phiG":48, "t(s)":">3600"}},
    'le450_15a.col':{"MSBCOL":{"phiG":35, "t(s)":3.7},
                     "IMA":{"phiG":40, "t(s)":1800}},
    'le450_15c.col':{"MSBCOL":{"phiG":41, "t(s)":6.4},
                     "IMA":{"phiG":54, "t(s)":1800}},
    'le450_25a.col':{"MSBCOL":{"phiG":41, "t(s)":4.8},
                     "IMA":{"phiG":54, "t(s)":1800}},
    'school1.col':{"MSBCOL":{"phiG":58, "t(s)":14.1},
                   "IMA":{"phiG":70, "t(s)":1800}}


}

results_pr = { "G":[], "V(G)":[], "E(G)":[], "DENSITY(G)":[], "m(G)":[],
               "PR1(G)":[], "TPR1(s)":[], "PR1_EXT1":[], "TPR1E1(s)":[],
               "PR1_EXT2":[], "TPR1E2(s)":[], "PR2(G)":[], "TPR2(s)":[],
               "PR2_EXT1":[], "TPR2E1(s)":[], "PR2_EXT2":[], "TPR2E2(s)":[],
               "PR3(G)":[], "TPR3(s)":[], "PR3_EXT1":[], "TPR3E1(s)":[],
               "PR3_EXT2":[], "TPR3E2(s)":[]}
for alg, talg in algorithms:
    results_pr[alg] = []
    results_pr[talg] = []

def read_graph(name):
    dir = "./graphs_datasets/DIMACS"
    file = "{}/{}"
    with open(file.format(dir, name), 'r') as f:
        while(True):
            line = f.readline()
            if line == "": break
            if line[0] == "p":
                _, _, n, m = line.split()
                n = int(n)
                m = int(m)
                G = nx.Graph()
                for i in range(1, n+1): G.add_node(i)
            elif line[0] == "e": 
                _, u, v = line.split()
                G.add_edge(int(u), int(v))
    return G

def run_experiment(N, filename):
    exp_start = time.time()
    for gname in graphs_list:
        print("Computing on graph: {}".format(gname), end="r")
        G = read_graph(gname)
        G = get_biggest_cc(G)
        results_pr["G"].append(gname)
        results_pr["V(G)"].append(G.number_of_nodes())
        results_pr["E(G)"].append(G.number_of_edges())
        results_pr["m(G)"].append(get_m_degree(G))
        results_pr["DENSITY(G)"].append(2*G.number_of_edges()/(G.number_of_nodes()*(G.number_of_nodes()-1)))
        
        for alg, talg in algorithms:
            if alg in other_papers_res[gname]:
                results_pr[alg].append(other_papers_res[gname][alg]["phiG"])
                results_pr[talg].append(other_papers_res[gname][alg]["t(s)"])
            else:
                results_pr[alg].append("-")
                results_pr[talg].append("-")

        f = {u:i+1 for i, u in enumerate(G)}
        #compute the best approximation value with PR1
        fr1, bestPhi, total_time = run_test(G, f, partition_redistribution1, N)
        results_pr["PR1(G)"].append(bestPhi)
        results_pr["TPR1(s)"].append(total_time)

        #Now we try to extend the best approximation value
        k, total_time = extend_colouring(G, fr1, extend_b_chromatic_colouring_by_one1)
        results_pr["PR1_EXT1"].append(k)
        results_pr["TPR1E1(s)"].append(total_time)

        #Now we try to extend the best approximation value
        k, total_time = extend_colouring(G, fr1, extend_b_chromatic_colouring_by_one2)
        results_pr["PR1_EXT2"].append(k)
        results_pr["TPR1E2(s)"].append(total_time)



        #compute the best approximation value with PR1
        fr2, bestPhi, total_time = run_test(G, f, partition_redistribution2, N)
        results_pr["PR2(G)"].append(bestPhi)
        results_pr["TPR2(s)"].append(total_time)

        #Now we try to extend the best approximation value
        k, total_time = extend_colouring(G, fr2, extend_b_chromatic_colouring_by_one1)
        results_pr["PR2_EXT1"].append(k)
        results_pr["TPR2E1(s)"].append(total_time)

        #Now we try to extend the best approximation value
        k, total_time = extend_colouring(G, fr2, extend_b_chromatic_colouring_by_one2)
        results_pr["PR2_EXT2"].append(k)
        results_pr["TPR2E2(s)"].append(total_time)

        #compute the best approximation value with PR1
        fr3, bestPhi, total_time = run_test(G, f, partition_redistribution3, N)
        results_pr["PR3(G)"].append(bestPhi)
        results_pr["TPR3(s)"].append(total_time)

        #Now we try to extend the best approximation value
        k, total_time = extend_colouring(G, fr3, extend_b_chromatic_colouring_by_one1)
        results_pr["PR3_EXT1"].append(k)
        results_pr["TPR3E1(s)"].append(total_time)

        #Now we try to extend the best approximation value
        k, total_time = extend_colouring(G, fr3, extend_b_chromatic_colouring_by_one2)
        results_pr["PR3_EXT2"].append(k)
        results_pr["TPR3E2(s)"].append(total_time)

    pd.DataFrame(results_pr).to_csv(filename)
    exp_end = time.time()
    print("Experiment finalized in {} seconds\n".format(exp_end-exp_start))

if __name__ == "__main__":
    N = int(sys.argv[1])
    now = datetime.now()
    folder = "./experiment_results/heuristics/partition_redistribution/{}"
    if os.path.exists(folder[:-3]) is False:
        os.mkdir(folder[:-3])
    filename = "{}_{}_{}_{}_{}_results.csv".format(now.day, now.month, now.year, now.hour, now.minute)
    run_experiment(N, folder.format(filename))