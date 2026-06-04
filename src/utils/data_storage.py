import pickle

def write_graph_into_txt(T, filename):
    file_graph = open(filename, "a")
    file_graph.write("{} {}\n".format(len(T.nodes()), len(T.edges())))
    for u, v in T.edges():
        file_graph.write("{} {}\n".format(u, v))
    file_graph.close()

def get_edges_lists(file, n_edges):
    edges_list = []
    for i in range(n_edges):
        edge = file.readline().split()
        #print(edge)
        edges_list.append([edge[0], edge[1]])
    return edges_list

def store_graphs_as_list_of_dicts(filename, graph_list):
    #dict_of_trees = {}
    #for key in graph_list.keys():
    #    list_of_dicts = [nx.to_dict_of_lists(G) for G in graph_list[key]]
        #dict_of_trees[key] = list_of_dicts
    list_of_dicts = [nx.to_dict_of_lists(G) for G in graph_list]
    with open(filename, 'wb') as f:
        #pickle.dump(dict_of_trees, f)
        pickle.dump(list_of_dicts, f)

def load_graph_from_list_of_dicts(filename):
    with open(filename, 'rb') as f:
    #    dict_of_trees = pickle.load(f)
        list_of_dicts = pickle.load(f)
    #graphs_dict = {}
    #for key in dict_of_trees.keys():
    #    graphs = [nx.from_dict_of_lists(G) for G in dict_of_trees[key]]
    #    graphs_dict[key] = graphs
    list_of_trees = [nx.from_dict_of_lists(G) for G in list_of_dicts]
    return list_of_trees