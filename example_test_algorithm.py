import numpy as np
import random
import time
import pickle

from src.instance_mcnf import generate_instance
from src.decomposition_methods import run_DW_Fenchel_model, knapsack_model_solver, compute_possible_paths_per_commodity, knapsack_cut_lowerbound, dijkstra

for i in range(1):
    # CHOOSE THE SETTING OF THE INSTANCES
    size = 30 # Size of the graph. Note that grid graphs and random connected graphs don't use the size parameter in the same way (see paper). For random connected graphs the size is the number of nodes
    arc_capacity = 1000 # Capacity of the arcs of the graph
    max_demand = 1000 # Upper bound on the size of the commodities

    path_generation_loop = False

    # Select the type of graph to create:
    graph_type = "random_connected"
    # graph_type = "grid"

    smaller_commodities = True # determines what formulae is used to create the demand of a commodity (see annex of the paper)

    # CHOOSE THE TESTED ALGORITHMS
    tested_algorithms = []
    # tested_algorithms.append("knapsack cut lowerbound")
    # tested_algorithms.append("DW")
    # tested_algorithms.append("DW momentum")
    tested_algorithms.append("DW interior")
    # tested_algorithms.append("DW in out")
    # tested_algorithms.append("Fenchel")
    # tested_algorithms.append("Fenchel no preprocessing")
    # tested_algorithms.append("DW-Fenchel")
    # tested_algorithms.append("DW-Fenchel no preprocessing")
    tested_algorithms.append("DW-Fenchel iterative")


    # Setting parameters for the instance generator
    if graph_type == "random_connected":
        graph_generator_inputs = (size, 5/size, int(size * 0.1), arc_capacity)
        nb_nodes = size
    elif graph_type == "grid":
        graph_generator_inputs = (size, size, size, 2*size, arc_capacity, arc_capacity)
        nb_nodes = size ** 2 + size
    demand_generator_inputs = {"max_demand" : max_demand, "smaller_commodities" : smaller_commodities}


    # Choice of the seed
    seed = random.randint(0, 10**5)
    # seed = 45440
    # seed = i
    print("seed = ", seed)
    random.seed(seed); np.random.seed(seed)

    # Instance generation
    graph, commodity_list, initial_solution, origin_list = generate_instance(graph_type, graph_generator_inputs, demand_generator_inputs, nb_capacity_modifitcations=0 * size)


    print("total_demand is : ", sum([commodity[2] for commodity in commodity_list]))
    print("nb_commodities = ", len(commodity_list))
    print("nb_nodes = ", len(graph))
    print("nb_arcs = ", sum(len(l) for l in graph))
    # print(commodity_list)

    # Computes a restricted set of paths to be used by each commodity
    possible_paths_per_commodity = compute_possible_paths_per_commodity(graph, commodity_list, 4)
    if not path_generation_loop:
        for commodity_index in range(len(commodity_list)):
            possible_paths_per_commodity[commodity_index].append(initial_solution[commodity_index])
    # possible_paths_per_commodity=None


    # possible_paths_per_commodity = [[path] for path in initial_solution]
    # for origin, destination, demand in commodity_list:

    #     for repetition in range(8):
    #         graph_weighted = [{neighbor : 1/random.random() for neighbor in neighbor_list} for neighbor_list in graph]
    #         path, distances = dijkstra(graph_weighted, origin, destination)
    #         if path not in possible_paths_per_commodity[commodity_index]:
    #             possible_paths_per_commodity[commodity_index].append(path)
    
    print(sum(len(l) for l in possible_paths_per_commodity) / len(commodity_list))

    # Applying the algorithms present in tested_algorithms
    for algorithm_name in tested_algorithms:
        print("Running {}".format(algorithm_name))
        temp = time.time()
        return_list = []
        verbose=1

        # import cProfile, pstats, io
        # from pstats import SortKey
        # pr = cProfile.Profile()
        # pr.enable()
        # ... do something ...

        if algorithm_name == "DW" : knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, stabilisation="", path_generation_loop=path_generation_loop, verbose=verbose)
        if algorithm_name == "DW momentum" : knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, stabilisation="momentum", path_generation_loop=path_generation_loop, verbose=verbose)
        if algorithm_name == "DW in out" : knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, stabilisation="in_out", path_generation_loop=path_generation_loop, verbose=verbose)
        if algorithm_name == "DW interior" : knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, stabilisation="interior_point", path_generation_loop=path_generation_loop, verbose=verbose)
        if algorithm_name == "Fenchel" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(False, True, False), path_generation_loop=path_generation_loop, verbose=verbose)
        if algorithm_name == "Fenchel no preprocessing" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(False, False, False), path_generation_loop=path_generation_loop, verbose=verbose)
        if algorithm_name == "DW-Fenchel" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(True, True, False), path_generation_loop=path_generation_loop, verbose=verbose)
        if algorithm_name == "DW-Fenchel no preprocessing" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(True, False, False), path_generation_loop=path_generation_loop, verbose=verbose)
        if algorithm_name == "DW-Fenchel iterative" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(True, True, True), path_generation_loop=path_generation_loop, verbose=verbose)
        if algorithm_name == "knapsack cut lowerbound" : knapsack_cut_lowerbound(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, verbose=verbose)

        # pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

        computing_time = time.time() - temp
        print("computing_time = ", computing_time)
