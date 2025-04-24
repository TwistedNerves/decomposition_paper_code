import numpy as np
import random
import time
import heapq as hp
import gurobipy

from src.knapsack_oracles import separation_decomposition_with_preprocessing


def create_arc_path_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=1, deviation_penalization=10**5, verbose=0):
    # creates a linear model for the linear relaxation of the unsplittable flow problem based on an arc-path formulation
    nb_commodities = len(commodity_list)
    nb_nodes = len(graph)
    demand_list = [commodity[2] for commodity in commodity_list]
    arc_list = [(node, neighbor) for node in range(nb_nodes) for neighbor in graph[node]]
    max_demand = max(demand_list)

    # Create optimization model
    model = gurobipy.Model('netflow')
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = verbose
    model.Params.Method = 3

    # Create variables
    path_and_var_per_commodity = [[(path, model.addVar(obj=(len(path) - 1) * flow_penalisation )) for path in possible_paths_per_commodity[commodity_index]] for commodity_index, demand in enumerate(demand_list)]
    deviation_variables = model.addVars(nb_commodities, obj=deviation_penalization)
    if verbose:
        print("variables created")

    # Convexity constraints :
    convexity_constraint_dict = model.addConstrs((sum(var for path, var in path_and_var_per_commodity[commodity_index]) == 1 - deviation_variables[commodity_index] for commodity_index in range(nb_commodities)))
    if verbose:
        print("Convexity constraints created")

    # Capacity constraint
    edge_var_sum_dict = {arc : 0 for arc in arc_list}
    for commodity_index, demand in enumerate(demand_list):
        for path, var in path_and_var_per_commodity[commodity_index]:
            for node_index in range(len(path)-1):
                arc = (path[node_index], path[node_index+1])
                edge_var_sum_dict[arc] += var * demand

    capacity_constraint_dict = model.addConstrs((edge_var_sum_dict[arc] <= graph[arc[0]][arc[1]] for arc in arc_list))
    if verbose:
        print("Capacity constraints created")

    model.update()
    return model, (path_and_var_per_commodity, deviation_variables), (convexity_constraint_dict, capacity_constraint_dict)



def create_knapsack_model(graph, commodity_list, possible_paths_per_commodity, deviation_penalization=10**5, flow_penalisation=1, verbose=1):
    # creates the linear model obtained after applying a Dantzig-Wolfe decomposition to the capacity constraints
    # of an arc-path formulation of the unsplittable flow problem
    nb_commodities = len(commodity_list)
    nb_nodes = len(graph)
    demand_list = [commodity[2] for commodity in commodity_list]
    arc_list = [(node, neighbor) for node in range(nb_nodes) for neighbor in graph[node]]

    # creates the model for an arc path formulation that will be modified
    print(flow_penalisation)
    model, variables, constraints = create_arc_path_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=flow_penalisation, verbose=0)
    path_and_var_per_commodity, deviation_variables = variables
    convexity_constraint_dict, capacity_constraint_dict = constraints

    # removing the uselles parts of the arc-path formualtion
    for constraint in capacity_constraint_dict.values():
        model.remove(constraint)


    # creating the inital pattern variables for each arc
    pattern_and_var_per_arc = {}
    knapsack_convexity_constraint_dict = {}
    for arc in arc_list:
        var = model.addVar()
        pattern_and_var_per_arc[arc] = [([], var)]
        knapsack_convexity_constraint_dict[arc] = model.addConstr(var <= 1)

    # constraints linking the flow variables and the pattern variables
    linking_constraint_dict = {arc : [None] * nb_commodities for arc in arc_list}
    for commodity_index, path_and_var in enumerate(path_and_var_per_commodity):
        edge_var_sum_dict = {}

        for path, var in path_and_var:
            for node_index in range(len(path)-1):
                arc = (path[node_index], path[node_index+1])
                if arc not in edge_var_sum_dict:
                    edge_var_sum_dict[arc] = 0
                edge_var_sum_dict[arc] += var

        for arc in edge_var_sum_dict:
            linking_constraint_dict[arc][commodity_index] = model.addConstr((edge_var_sum_dict[arc] <= 0 ), "capacity")

    if verbose: print("Linking constraints created")

    model.update()
    return model, (path_and_var_per_commodity, pattern_and_var_per_arc, deviation_variables), (convexity_constraint_dict, knapsack_convexity_constraint_dict, linking_constraint_dict)



def create_arc_node_model(graph, commodity_list, flow_penalisation=1, deviation_penalization=10**5, verbose=0):
    nb_nodes = len(graph)
    nb_commodities = len(commodity_list)
    arc_list = [(node, neighbor) for node in range(nb_nodes) for neighbor in graph[node]]
    demand_list = [demand for origin, destination, demand in commodity_list]
    max_demand = max(demand_list)

    # Create optimization model
    model = gurobipy.Model('netflow')
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = verbose>1

    # Create variables
    flow_variables = [model.addVars(arc_list, obj=flow_penalisation ) for demand in demand_list] # flow variables
    deviation_variables = model.addVars(nb_commodities, obj=deviation_penalization)

    # Arc capacity constraints :
    capacity_constraint_dict = {}
    for arc in arc_list:
        flow_on_arc = sum(flow_variables[commodity_index][arc] * demand_list[commodity_index] for commodity_index in range(nb_commodities))
        capacity_constraint_dict[arc] = model.addConstr(flow_on_arc <= graph[arc[0]][arc[1]])

    # Flow conservation constraints
    flow_constraint_dict = {node : [] for node in range(nb_nodes)}
    for node in range(nb_nodes):
        for commodity_index, commodity in enumerate(commodity_list):
            origin, destination, demand = commodity
            rhs = ((node == origin) - (node == destination)) * (1 - deviation_variables[commodity_index])
            constraint = model.addConstr(flow_variables[commodity_index].sum(node,'*') - flow_variables[commodity_index].sum('*',node) == rhs)
            flow_constraint_dict[node].append(constraint)

    model.update()

    return model, (deviation_variables, flow_variables), (capacity_constraint_dict, flow_constraint_dict)


if __name__ == "__main__":
    pass
