import matplotlib.pyplot as plt
import numpy as np
import random
import time

from src.instance_mcnf import generate_instance
from src.decomposition_methods import run_DW_Fenchel_model, knapsack_model_solver, compute_possible_paths_per_commodity

# CHOOSE THE SETTING OF THE INSTANCES
size = 50 # Size of the graph. Note that grid graphs and random connected graphs don't use the size parameter in the same way (see paper). For random connected graphs the size is the number of nodes
arc_capacity = 1000 # Capacity of the arcs of the graph
max_demand = 100 # Upper bound on the size of the commodities

# Select the type of graph to create:
graph_type = "random_connected"
# graph_type = "grid"

smaller_commodities = True # determines what formulae is used to create the demand of a commodity (see annex of the paper)

# CHOOSE THE TESTED ALGORITHMS
tested_algorithms = []
# tested_algorithms.append("DW")
# tested_algorithms.append("DW momentum")
# tested_algorithms.append("DW interior")
# tested_algorithms.append("Fenchel")
# tested_algorithms.append("Fenchel no preprocessing")
tested_algorithms.append("DW-Fenchel")
# tested_algorithms.append("DW-Fenchel no preprocessing")
# tested_algorithms.append("DW-Fenchel iterative")


# Setting parameters for the instance generator
if graph_type == "random_connected":
    graph_generator_inputs = (size, 5/size, int(size * 0.1), arc_capacity)
    nb_nodes = size
elif graph_type == "grid":
    graph_generator_inputs = (size, size, size, 2*size, arc_capacity, arc_capacity)
    nb_nodes = size ** 2 + size
demand_generator_inputs = {"max_demand" : max_demand}


# Choice of the seed
seed = random.randint(0, 10**5)
# seed = 72565
print("seed = ", seed)
random.seed(seed); np.random.seed(seed)

# Instance generation
graph, commodity_list, initial_solution, origin_list = generate_instance(graph_type, graph_generator_inputs, demand_generator_inputs, nb_capacity_modifitcations=100 * size)

print("total_demand is : ", sum([commodity[2] for commodity in commodity_list]))
print("nb_commodities = ", len(commodity_list))
print("nb_nodes = ", len(graph))

# Computes a restricted set of paths to be used by each commodity
possible_paths_per_commodity = compute_possible_paths_per_commodity(graph, commodity_list, 4)
for commodity_index in range(len(commodity_list)):
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
