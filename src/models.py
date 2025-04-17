import numpy as np
import random
import time
import heapq as hp
import gurobipy

from src.knapsack_oracles import separation_decomposition_with_preprocessing


def create_arc_path_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=1, overload_penalization=10**5, verbose=0):
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
    path_and_var_per_commodity = [[(path, model.addVar(obj=(len(path) - 1) * flow_penalisation * (demand / max_demand * 0 + 1))) for path in possible_paths_per_commodity[commodity_index]] for commodity_index, demand in enumerate(demand_list)]
    overload_variables = model.addVars(arc_list, obj=overload_penalization, name="overload") # overload variables : we want to minimize their sum
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

    capacity_constraint_dict = model.addConstrs((edge_var_sum_dict[arc] - overload_variables[arc] <= graph[arc[0]][arc[1]] for arc in arc_list))
    if verbose:
        print("Capacity constraints created")

    model.update()
    return model, (path_and_var_per_commodity, overload_variables), (convexity_constraint_dict, capacity_constraint_dict)



def create_knapsack_model(graph, commodity_list, possible_paths_per_commodity, overload_penalization=10**5, flow_penalisation=1, verbose=1):
    # creates the linear model obtained after applying a Dantzig-Wolfe decomposition to the capacity constraints
    # of an arc-path formulation of the unsplittable flow problem
    nb_commodities = len(commodity_list)
    nb_nodes = len(graph)
    demand_list = [commodity[2] for commodity in commodity_list]
    arc_list = [(node, neighbor) for node in range(nb_nodes) for neighbor in graph[node]]

    # creates the model for an arc path formulation that will be modified
    print(flow_penalisation)
    model, variables, constraints = create_arc_path_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=flow_penalisation, verbose=0)
    path_and_var_per_commodity, overload_variables = variables
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
        _, pattern_overload_and_amount_list = separation_decomposition_with_preprocessing(demand_list, flow_per_commodity_per_arc[arc], arc_capacity)
        pattern_and_cost_per_arc[arc] = [(pattern, pattern_overload) for pattern, pattern_overload, amount in pattern_overload_and_amount_list]


    # removing the uselles parts of the arc-path formualtion
    for constraint in capacity_constraint_dict.values():
        model.remove(constraint)

    for var in overload_variables.values():
        model.remove(var)

    # creating the inital pattern variables for each arc
    pattern_var_and_cost_per_arc = {}
    knapsack_convexity_constraint_dict = {}
    for arc in arc_list:
        pattern_var_and_cost_per_arc[arc] = []

        for pattern, pattern_overload in pattern_and_cost_per_arc[arc]:
            pattern_var_and_cost_per_arc[arc].append((pattern, model.addVar(obj=overload_penalization * pattern_overload), pattern_overload))

        knapsack_convexity_constraint_dict[arc] = model.addConstr(sum(var for pattern, var, pattern_overload in pattern_var_and_cost_per_arc[arc]) <= 1)

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
            knapsack_var_sum = sum(var for pattern, var, pattern_overload in pattern_var_and_cost_per_arc[arc] if commodity_index in pattern)
            linking_constraint_dict[arc][commodity_index] = model.addConstr((edge_var_sum_dict[arc] - knapsack_var_sum <= 0 ), "capacity")

    if verbose: print("Linking constraints created")

    model.update()
    return model, (path_and_var_per_commodity, pattern_var_and_cost_per_arc), (convexity_constraint_dict, knapsack_convexity_constraint_dict, linking_constraint_dict)



def create_arc_node_model(graph, commodity_list, flow_penalisation=1, overload_penalization=10**5, verbose=0):
    # LP program that solves the multicommodity flow problem with the following objective function : minimize the sum of the arc_list overload
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
    flow_variables = [model.addVars(arc_list, obj=flow_penalisation * (demand / max_demand * 0 + 1)) for demand in demand_list] # flow variables
    overload_variables = model.addVars(arc_list, obj=overload_penalization) # overload variables : we want to minimize their sum

    # Arc capacity constraints :
    capacity_constraint_dict = {}
    for arc in arc_list:
        flow_on_arc = sum(flow_variables[commodity_index][arc] * demand_list[commodity_index] for commodity_index in range(nb_commodities))
        capacity_constraint_dict[arc] = model.addConstr(flow_on_arc <= graph[arc[0]][arc[1]] + overload_variables[arc])

    # Flow conservation constraints
    flow_constraint_dict = {node : [] for node in range(nb_nodes)}
    for node in range(nb_nodes):
        for commodity_index, commodity in enumerate(commodity_list):
            origin, destination, demand = commodity
            rhs = (node == origin) - (node == destination)
            constraint = model.addConstr(flow_variables[commodity_index].sum(node,'*') - flow_variables[commodity_index].sum('*',node) == rhs)
            flow_constraint_dict[node].append(constraint)

    model.update()

    return model, (overload_variables, flow_variables), (capacity_constraint_dict, flow_constraint_dict)


if __name__ == "__main__":
    pass
