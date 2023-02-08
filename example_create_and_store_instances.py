import random
import numpy as np
import time
import pickle

from src.instance import generate_instance

# SET THE PARAMETERS OF THE DATASET HERE
nb_repetitions = 10 # number of instances with the same parameters
nb_unique_exp = 3 # number of different types of instances

# Size of the graph : controls the number of nodes and arcs
size_list = [145, 250, 400, 650, 1000]
# size_list = [70]*nb_unique_exp

# Capacity of the arcs of the graph
capacity_list = [1000] * nb_unique_exp
# capacity_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# Upper bound on the size of the commodities
max_demand_list = [1000] * nb_unique_exp
# max_demand_list = [int(np.sqrt(capacity)) for capacity in capacity_list]

# Select the type of graph to create: note that grid graphs and random connected graphs dont use the size parameter in the same way
instance_parameter_list = []
for size, capacity, max_demand in zip(size_list, capacity_list, max_demand_list):
    # instance_parameter_list.append(("grid", (size, size, size, 2*size, capacity, capacity), {"max_demand" : max_demand, "smaller_commodities" : False}))
    # instance_parameter_list.append(("grid", (size, size, size, 2*size, capacity, capacity), {"max_demand" : max_demand, "smaller_commodities" : True}))
    instance_parameter_list += [("random_connected", (size, 5/size, int(size * 0.1), capacity), {"max_demand" : max_demand, "smaller_commodities" : False})]

# Set the path to the repository folder
global_path = "/home/francois/Desktop/"
# assert False, "Unassigned global_path : Complete global_path with the path to the main directory"

# Complete name of the directory that will contain the instances
dataset_name = "graph_scaling_dataset_lower_bound/"
# dataset_name = "graph_scaling_dataset/"
# dataset_name = "small_dataset/"
# dataset_name = "small_dataset_lower_bound/"
# dataset_name = "capacity_scaling_dataset/"

instance_name_list = []
for graph_type, graph_generator_inputs, demand_generator_inputs in instance_parameter_list:
    for repetition_index in range(nb_repetitions):

        max_demand = demand_generator_inputs["max_demand"]
        if graph_type == "grid":
            size, _, _, _, capacity, _ = graph_generator_inputs
            nb_nodes = size ** 2 + size
        if graph_type == "random_connected":
            nb_nodes, _, _, capacity = graph_generator_inputs

        # Generate the graph and the commodity list
        graph, commodity_list, initial_solution, origin_list = generate_instance(graph_type, graph_generator_inputs, demand_generator_inputs, nb_capacity_modifitcations=100*nb_nodes)

        instance_name = graph_type + "_" + str(nb_nodes) + "_" + str(capacity) + "_" + str(max_demand) + "_" + str(repetition_index)

        # Store the created instance
        instance_file = open(global_path + "MCNF_solver/instance_files_decomposition/" + dataset_name + instance_name + ".p", "wb" )
        pickle.dump((graph, commodity_list, initial_solution), instance_file)
        instance_file.close()

        instance_name_list.append(instance_name)

# Create a file containing the name of all the instances
instance_name_file = open(global_path + "MCNF_solver/instance_files_decomposition/" + dataset_name + "instance_name_file.p", "wb" )
pickle.dump(instance_name_list, instance_name_file)
instance_name_file.close()
