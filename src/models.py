import numpy as np
import random
import time
import heapq as hp
import gurobipy

from src.knapsack_oracles import separation_decomposition_with_preprocessing


def create_arc_path_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=0, verbose=0):
    # creates a linear model for the linear relaxation of the unsplittable flow problem based on an arc-path formulation
    nb_commodities = len(commodity_list)
    nb_nodes = len(graph)
    demand_list = [commodity[2] for commodity in commodity_list]
    arc_list = [(node, neighbor) for node in range(nb_nodes) for neighbor in graph[node]]

    # Create optimization model
    model = gurobipy.Model('netflow')
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = verbose
    model.Params.Method = 3

    # Create variables
    path_and_var_per_commodity = [[(path, model.addVar(obj=(len(path) - 1) * flow_penalisation)) for path in possible_paths] for possible_paths in possible_paths_per_commodity]
    overload_var = model.addVars(arc_list, obj=1, name="overload") # overload variables : we want to minimize their sum
    if verbose:
        print("variables created")

    # Convexity constraints :
    convexity_constraint_dict = model.addConstrs((sum(var for path, var in path_and_var_per_commodity[commodity_index]) == 1 for commodity_index in range(nb_commodities)))
    if verbose:
        print("Convexity constraints created")

    # Capacity constraint
    edge_var_sum_dict = {arc : 0 for arc in arc_list}
    for commodity_index, demand in enumerate(demand_list):
        for path, var in path_and_var_per_commodity[commodity_index]:
            for node_index in range(len(path)-1):
                arc = (path[node_index], path[node_index+1])
                edge_var_sum_dict[arc] += var * demand

    capacity_constraint_dict = model.addConstrs((edge_var_sum_dict[arc] - overload_var[arc] <= graph[arc[0]][arc[1]] for arc in arc_list))
    if verbose:
        print("Capacity constraints created")

    model.update()
    return model, (path_and_var_per_commodity, overload_var), (convexity_constraint_dict, capacity_constraint_dict)



def create_knapsack_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=0, verbose=1):
    # creates the linear model obtained after applying a Dantzig-Wolfe decomposition to the capacity constraints
    # of an arc-path formulation of the unsplittable flow problem
    nb_commodities = len(commodity_list)
    nb_nodes = len(graph)
    demand_list = [commodity[2] for commodity in commodity_list]
    arc_list = [(node, neighbor) for node in range(nb_nodes) for neighbor in graph[node]]

    # creates the model for an arc path formulation that will be modified
    model, variables, constraints = create_arc_path_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=flow_penalisation, verbose=0)
    path_and_var_per_commodity, overload_var = variables
    convexity_constraint_dict, capacity_constraint_dict = constraints

    # obtaining a solution of the arc-path formualtion enables us to create a set of valid variable for the Dantzig-Wolfe model
    model.update()
    model.optimize()
    if verbose : print("continuous ObjVal = ", model.ObjVal)

    flow_per_commodity_per_arc = {(arc): [0]*nb_commodities for arc in arc_list}
    for commodity_index, path_and_var in enumerate(path_and_var_per_commodity):
        for path, var in path_and_var:
            for node_index in range(len(path)-1):
                arc = (path[node_index], path[node_index + 1])
                flow_per_commodity_per_arc[arc][commodity_index] += var.X

    # the flow on each arc is decomposed into a convex combination of commodity patterns
    # which will become the set of pattern variables initially allowed on this arc
    pattern_and_cost_per_arc = {}
    for arc in arc_list:
        arc_capacity = graph[arc[0]][arc[1]]
        _, pattern_cost_and_amount_list = separation_decomposition_with_preprocessing(demand_list, flow_per_commodity_per_arc[arc], arc_capacity)
        pattern_and_cost_per_arc[arc] = [(pattern, pattern_cost) for pattern, pattern_cost, amount in pattern_cost_and_amount_list]
        pattern_and_cost_per_arc[arc].append((list(range(nb_commodities)), sum(demand_list) - arc_capacity))

    # removing the uselles parts of the arc-path formualtion
    for constraint in capacity_constraint_dict.values():
        model.remove(constraint)

    for var in overload_var.values():
        model.remove(var)

    # creating the inital pattern variables for each arc
    pattern_var_and_cost_per_arc = {}
    knapsack_convexity_constraint_dict = {}
    for arc in arc_list:
        pattern_var_and_cost_per_arc[arc] = []

        for pattern, pattern_cost in pattern_and_cost_per_arc[arc]:
            pattern_var_and_cost_per_arc[arc].append((pattern, model.addVar(obj=pattern_cost), pattern_cost))

        knapsack_convexity_constraint_dict[arc] = model.addConstr(sum(var for pattern, var, pattern_cost in pattern_var_and_cost_per_arc[arc]) <= 1)

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
            knapsack_var_sum = sum(var for pattern, var, pattern_cost in pattern_var_and_cost_per_arc[arc] if commodity_index in pattern)
            linking_constraint_dict[arc][commodity_index] = model.addConstr((edge_var_sum_dict[arc] - knapsack_var_sum <= 0 ), "capacity")

    if verbose: print("Linking constraints created")

    model.update()
    return model, (path_and_var_per_commodity, pattern_var_and_cost_per_arc), (convexity_constraint_dict, knapsack_convexity_constraint_dict, linking_constraint_dict)



if __name__ == "__main__":
    pass
