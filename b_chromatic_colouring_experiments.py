from logging.config import dictConfig
from timeit import default_timer as timer
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from b_chromatic_colouring import *
from b_chromatic_utils import *

def plot_lines(xs, ys, labels, x_label, y_label, filename):
    fig, ax = plt.subplots(1)
    for x, y, l in zip(xs, ys, labels):
        x = np.array(x)
        y = np.array(y)
        ax.plot(x, y, label=l)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    fig.savefig(filename)

    
def scatter_plot(xs, ys, labels, x_label, y_label, filename):
    fig, ax = plt.subplots(1)
    for x, y, l in zip(xs,ys, labels):
        x = np.array(x)
        y = np.array(y)
        ax.scatter(x, y, label=l)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    fig.savefig(filename)


def generate_random_trees(n_vertices_list, n_repetitions, f_tree, filename):
    with open(filename, "a") as f:
        for n in n_vertices_list:
            for _ in range(n_repetitions):
                Ti = f_tree(n)
                f.write("{} {}\n".format(len(Ti.nodes()), len(Ti.edges())))
                for u, v in Ti.edges():
                    f.write("{} {}\n".format(u,v))


def compute_statistics_on_random_trees(filename, tree_filename=None, 
        n_vertices_increment=None, max_n_vertices=None,initial_n=None):
    print("Loading {} samples for computing statistics".format(filename.replace("_", " ")))
    #trees_dict = load_graph_from_list_of_dicts(filename)
    #n_vertices_increment = list(trees_dict.keys())[1] - list(trees_dict.keys())[0]
    #max_n_vertices = max(trees_dict.keys()) + n_vertices_increment
    n_vertices_list = [i for i in range(initial_n, max_n_vertices, n_vertices_increment)]
    statistics_list = ["max_degree_avg", "m_degree_avg", "dense_vertices_avg", "non_pivoted_avg", "b_chromatic_avg", "colouring_time_avg"]
    dict_statistics = {stats:[] for stats in statistics_list}
    dict_statistics["n_vertices_list"] = n_vertices_list
    for n in n_vertices_list:
        trees_list = load_graph_from_list_of_dicts(filename.format(tree_name, n))
        dict_averages = {stats:0 for stats in statistics_list}
        aux = {}
        for Ti in trees_list:
            start_colouring = timer()
            aux["max_degree_avg"], aux["m_degree_avg"], aux["dense_vertices_avg"], aux["b_chromatic_avg"], aux["non_pivoted_avg"], _, _ = colour_tree(Ti)
            end_colouring = timer()
            aux["colouring_time_avg"] = end_colouring-start_colouring
            for stats in statistics_list:
                dict_averages[stats] += aux[stats]
        for stats in statistics_list:
            dict_statistics[stats].append(dict_averages[stats]/n)
       
    return dict_statistics, statistics_list

def run_verification(filename, tree_name=None, n_vertices_increment=None, max_n_vertices=None, initial_n=None):
    print("Loading {} samples for verification".format(filename.replace("_", " ")))
    #trees_dict = load_graph_from_list_of_dicts(filename)
    n_vertices_list = [i for i in range(initial_n, max_n_vertices, n_vertices_increment)]
    for n in n_vertices_list:
        trees_list = load_graph_from_list_of_dicts(filename.format(tree_name, n))
        for Ti in trees_list:
            _, m_degree, _, b_chromatic, non_pivoted, colors, W = colour_tree(Ti)
            #VERIFICATION 1
            if non_pivoted == 1:
                assert b_chromatic == m_degree
            else:
                assert b_chromatic == m_degree - 1
            #VERIFICATION 2
            m_degree2 = get_m_degree_2(Ti)
            assert m_degree == m_degree2
            #VERIFICATION 3
            check_b_chromatic_coloring(Ti, W, colors, m_degree)

if __name__ == "__main__":
    #EXECUTION 1: python3 b_chromatic_colouring_experiments.py n_attemps inintial_n max_n_vertices n_vertices_increment
    #EXECUTION 2: python3 b_chromatic_colouring_experiments.py experiment_folder
    dataset_folder = "{}/datasets/{}"
    plots_folder = "{}/plots/{}"
    generate_dateset = False
    if len(sys.argv) == 2:
        parent_directory = sys.argv[1]
    else:
        n_attempts = int(sys.argv[1])
        initial_n = int(sys.argv[2])
        max_n_vertices = int(sys.argv[3])
        n_vertices_increment = int(sys.argv[4])
        parent_directory = "experiment_{}".format(datetime.now().strftime("%H_%M_%d_%m_%Y"))
        os.mkdir(parent_directory)
        os.mkdir(plots_folder[:-3].format(parent_directory))
        os.mkdir(dataset_folder[:-3].format(parent_directory))
        generate_dateset = True
    results_dict = {}
    random_tree_generators = {  "Random recursive tree":random_recursive_tree, 
                                "MST tree":random_mst_tree, 
                                "BFS tree":random_bfs_tree, 
                                "Pruffer tree":nx.random_tree
                            }
    filename = dataset_folder.format(parent_directory, "{}_{}.pkl")
    #filename = dataset_folder.format(parent_directory, "{}.pkl")

    if generate_dateset:    
        generate_random_trees(n_attempts, max_n_vertices, n_vertices_increment, 
            random_tree_generators, filename, initial_n)
    for name, f_name in random_tree_generators.items():
        tree_name = name.replace(" ","_")
        dict_statistics, statistics_list= compute_statistics_on_random_trees(filename, 
            tree_name, n_vertices_increment, max_n_vertices, initial_n)
        #dict_statistics, statistics_list= compute_statistics_on_random_trees(filename.format(tree_name))
        run_verification(filename, tree_name, n_vertices_increment, 
            max_n_vertices, initial_n)
        #run_verification(filename.format(tree_name))
        results_dict[name] = dict_statistics
    statistics_result_lists = {stats_name:None for stats_name in statistics_list}
    statistics_result_lists["n_vertices_list"] = [results_dict[name]["n_vertices_list"] for name in results_dict.keys()]
    for stats_name in statistics_list:
        statistics_result_lists[stats_name] = [results_dict[name][stats_name] for name in results_dict.keys()]
   

    plot_lines(statistics_result_lists["n_vertices_list"], statistics_result_lists["m_degree_avg"], random_tree_generators.keys(),\
                 "number of vertices", "m-degree", plots_folder.format(parent_directory, "n_vertices_v_m_degree"))
    
    plot_lines(statistics_result_lists["n_vertices_list"], statistics_result_lists["dense_vertices_avg"],\
                random_tree_generators.keys(), "number of vertices", "avg number of dense vertices",\
                plots_folder.format(parent_directory, "n_vertices_v_dense_vertices"))
   
    plot_lines(statistics_result_lists["n_vertices_list"], statistics_result_lists["non_pivoted_avg"],\
                random_tree_generators.keys(), "number of vertices", "proportion of non pivoted trees",\
                plots_folder.format(parent_directory, "n_vertices_v_non_pivoted_trees"))
   
    plot_lines(statistics_result_lists["n_vertices_list"], statistics_result_lists["max_degree_avg"],\
                random_tree_generators.keys(), "number of vertices", "max degree",\
                plots_folder.format(parent_directory, "n_vertices_v_max_degree"))
    
    plot_lines(statistics_result_lists["dense_vertices_avg"], statistics_result_lists["m_degree_avg"],\
                 random_tree_generators.keys(), "number of dense vertices", "m-degree",\
                 plots_folder.format(parent_directory, "dense_vertices_v_m_degree"))
    
    plot_lines(statistics_result_lists["n_vertices_list"], statistics_result_lists["b_chromatic_avg"],\
                 random_tree_generators.keys(), "number of vertices", "avg b-chromatic number",\
                 plots_folder.format(parent_directory, "n_vertices_v_b_chromatic_number"))

    plot_lines(statistics_result_lists["n_vertices_list"], statistics_result_lists["colouring_time_avg"],\
                 random_tree_generators.keys(), "number of vertices", "avg time to colour",\
                 plots_folder.format(parent_directory, "n_vertices_v_time_to_colour"))  