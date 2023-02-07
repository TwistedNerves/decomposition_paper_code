import matplotlib.pyplot as plt
import numpy as np
import random
import time

from instance_mcnf import generate_instance
from decomposition_methods import run_DW_Fenchel_model, knapsack_model_solver, compute_possible_paths_per_commodity

# CHOOSE THE SETTING OF THE INSTANCES

# Size of the graph
size_list = [90]*100
# size_list = [3, 4, 5, 6, 7, 9, 10, 12, 13, 15]
# size_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20]
# size_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20]
# size_list = [30, 50, 70, 100, 130, 160, 200, 250, 300, 400]
size_list = np.array(size_list)

# Capacity of the arcs of the graph
# capacity_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
capacity_list = [1000] * len(size_list)
# capacity_list = [10000] * len(size_list)
# capacity_list = [3] * len(size_list)


# Upper bound on the size of the commodities
# max_demand_list = [10, 20 , 50, 100, 200, 500, 1000, 2000, 5000, 10000]
max_demand_list = [100] * len(size_list)
# max_demand_list = [1500] * len(size_list)
# max_demand_list = [2] * len(size_list)
# max_demand_list = [capa / 5 for capa in capacity_list]
# max_demand_list = [int(np.sqrt(capa)) for capa in capacity_list]

# Select the type of graph to create: note that grid graphs and random connected graphs dont use the size parameter in the same way (see paper). For random connected graphs the size is the number of nodes
test_list = []
for size, capacity, max_demand in zip(size_list, capacity_list, max_demand_list):
    # test_list += [("grid", (size, size, size, 2*size, capacity, capacity), {"max_demand" : max_demand, "smaller_commodities" : False})]
    # test_list += [("grid", (size, size, size, 2*size, capacity, capacity), {"max_demand" : max_demand, "smaller_commodities" : True})]
    # test_list += [("random_connected", (size, 5/size, int(size * 0.1), capacity), {"max_demand" : max_demand, "smaller_commodities" : False})]
    test_list += [("random_connected", (size, 5/size, int(size * 0.1), capacity), {"max_demand" : max_demand, "smaller_commodities" : True})]


# CHOOSE THE TESTED ALGORITHMS
tested_algorithms = []
# tested_algorithms.append("DW")
# tested_algorithms.append("DW momentum")
tested_algorithms.append("DW interior")
# tested_algorithms.append("Fenchel")
# tested_algorithms.append("Fenchel no preprocessing")
# tested_algorithms.append("DW-Fenchel")
# tested_algorithms.append("DW-Fenchel no preprocessing")
# tested_algorithms.append("DW-Fenchel iterative")

results_dict = {algorithm_name : ([],[]) for algorithm_name in tested_algorithms}

i = -1
nb_commodity_list = []
nb_node_list = []

for graph_type, graph_generator_inputs, demand_generator_inputs in test_list:
    i += 1
    print("##############################  ", i,"/",len(test_list))

    # Choice of the seed
    seed = random.randint(0, 10**5)
    # seed = 72565
    print("seed = ", seed)
    random.seed(seed)
    np.random.seed(seed)

    # Instance generation
    nb_nodes = size_list[0]
    graph, commodity_list, initial_solution, origin_list = generate_instance(graph_type, graph_generator_inputs, demand_generator_inputs, nb_capacity_modifitcations=100 * nb_nodes)

    total_demand = sum([c[2] for c in commodity_list])
    nb_nodes = len(graph)
    nb_commodities = len(commodity_list)
    print("total_demand is : ", total_demand)
    print("nb_commodities = ", nb_commodities)
    print("nb_nodes = ", nb_nodes)
    nb_commodity_list.append(len(commodity_list))
    nb_node_list.append(nb_nodes)

    possible_paths_per_commodity = compute_possible_paths_per_commodity(graph, commodity_list, 4)
    for commodity_index in range(nb_commodities):
        possible_paths_per_commodity[commodity_index].append(initial_solution[commodity_index])

    # Applying the algorithms present in tested_algorithms
    for algorithm_name in tested_algorithms:
        print("Running {}".format(algorithm_name))
        temp = time.time()
        return_list = []
        verbose=1

        if algorithm_name == "DW" : knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, stabilisation="", verbose=verbose)
        if algorithm_name == "DW momentum" : knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, stabilisation="momentum", verbose=verbose)
        if algorithm_name == "DW interior" : knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, stabilisation="interior_point", verbose=verbose)
        if algorithm_name == "Fenchel" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(False, True, False), verbose=verbose)
        if algorithm_name == "Fenchel no preprocessing" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(False, False, False), verbose=verbose)
        if algorithm_name == "DW-Fenchel" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(True, True, False), verbose=verbose)
        if algorithm_name == "DW-Fenchel no preprocessing" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(True, False, False), verbose=verbose)
        if algorithm_name == "DW-Fenchel iterative" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(True, True, True), verbose=verbose)

        computing_time = time.time() - temp
        print("computing_time = ", computing_time)
