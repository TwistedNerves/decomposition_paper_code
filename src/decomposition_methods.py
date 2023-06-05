import numpy as np
import random
import time
import heapq as hp
import gurobipy

from src.k_shortest_path import k_shortest_path_all_destination
from src.models import create_arc_path_model, create_knapsack_model
from src.knapsack_oracles import penalized_knapsack_optimizer, in_out_separation_decomposition_with_preprocessing, in_out_separation_decomposition, separation_decomposition_with_preprocessing, separation_decomposition


def knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=None, nb_initial_path_created=4, var_delete_proba=0.3,
                            flow_penalisation=0, nb_iterations=10**5, bounds_and_time_list=[], stabilisation="interior_point", verbose=1):
    """
    Creates a knapsack model for the unsplittable flow problem (this model is the result of a Dantzig-Wolfe decompostion applied to the capacity contraints, see src.models.create_knapsack_model) and solves it with column generation

    Inputs:
    graph : graph of the unsplittable flow instance
    commodity_list : commodity list of the unsplittable flow instance
    possible_paths_per_commodity=None : a list of allowed paths for each commodity, if None the 4 k-shortest paths are generated
    stabilisation="interior_point" : determines which stabilizations is used for the column generation process, can be "interior_point", "momentum" or ""
    ...

    Outputs: None
    """
    nb_nodes = len(graph)
    nb_commodities = len(commodity_list)
    arc_list = [(node, neighbor) for node in range(len(graph)) for neighbor in graph[node]]
    demand_list = [demand for origin, destination, demand in commodity_list]

    if possible_paths_per_commodity is None:
        possible_paths_per_commodity = compute_possible_paths_per_commodity(graph, commodity_list, nb_initial_path_created)

    model, variables, constraints = create_knapsack_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=flow_penalisation, verbose=verbose>1)

    run_knapsack_model(graph, commodity_list, model, variables, constraints, stabilisation, bounds_and_time_list=bounds_and_time_list, verbose=verbose)


def run_knapsack_model(graph, commodity_list, model, variables, constraints, stabilisation, bounds_and_time_list=[], nb_iterations=10**5, initial_BarConvTol=10**-3, var_delete_proba=0.3, verbose=1):
    # column generation process used to solve the linear relaxation of a knapsack model (see create_knapsack_model) of the unsplittable flow problem
    nb_commodities = len(commodity_list)
    demand_list = [commodity[2] for commodity in commodity_list]
    arc_list = [(node, neighbor) for node in range(len(graph)) for neighbor in graph[node]]
    convexity_constraint_dict, knapsack_convexity_constraint_dict, capacity_constraint_dict = constraints
    path_and_var_per_commodity, pattern_var_and_cost_per_arc = variables
    starting_time = time.time()
    added_var_list = []
    nb_var_added = 0
    best_dual_bound = None
    used_dual_var_list_per_arc = None

    # parameter of the solver gurobi
    model.Params.Method = 3
    model.Params.OutputFlag = 0
    if stabilisation == "interior_point": # in this stabilisation, the master model is solved approximatly (10**-3 precision) with an interior point method
        model.update()
        model.optimize()
        model.Params.Method = 2
        model.Params.BarConvTol = initial_BarConvTol # precision of the interior point method
        model.Params.Crossover = 0

    for iter_index in range(nb_iterations):
        if verbose : print("iteration : ", iter_index)

        model.update()
        model.optimize()
        if verbose : print("Objective function value : ", model.ObjVal)
        if verbose : print("Master model runtime : ", model.Runtime)


        # variable deletion in the Dantzig-Wolfe model to prevent it from becoming to heavy
        if stabilisation != "interior_point":
            for arc in arc_list:
                l = []
                for pattern, var, pattern_cost in pattern_var_and_cost_per_arc[arc]:
                    if var.Vbasis != 0 and random.random() < var_delete_proba:
                        model.remove(var)
                    else:
                        l.append((pattern, var, pattern_cost))
                pattern_var_and_cost_per_arc[arc] = l

        # getting the dual variables of the master model
        dual_var_knapsack_convexity_per_arc = {arc : -knapsack_convexity_constraint_dict[arc].Pi for arc in arc_list}
        dual_var_flow_convexity_per_commoditiy = np.array([convexity_constraint_dict[commodity_index].Pi for commodity_index in range(nb_commodities)])
        dual_var_list_per_arc = {arc : np.array([-constraint.Pi if constraint is not None else 0 for constraint in capacity_constraint_dict[arc]]) for arc in arc_list}


        if stabilisation == "momentum" and used_dual_var_list_per_arc is not None: # another stabilisation, dual variables are aggregated through the iterations in a momentum fashion
            momentum_coeff = 0.8
            used_dual_var_list_per_arc = {arc : momentum_coeff * used_dual_var_list_per_arc[arc] + (1 - momentum_coeff) * dual_var_list_per_arc[arc] for arc in arc_list}
            used_dual_var_knapsack_convexity_per_arc = {arc : momentum_coeff * used_dual_var_knapsack_convexity_per_arc[arc] + (1 - momentum_coeff) * dual_var_knapsack_convexity_per_arc[arc] for arc in arc_list}
            used_dual_var_flow_convexity_per_commoditiy = momentum_coeff * used_dual_var_flow_convexity_per_commoditiy + (1 - momentum_coeff) * dual_var_flow_convexity_per_commoditiy

        else:
            used_dual_var_list_per_arc = dual_var_list_per_arc
            used_dual_var_knapsack_convexity_per_arc = dual_var_knapsack_convexity_per_arc
            used_dual_var_flow_convexity_per_commoditiy = dual_var_flow_convexity_per_commoditiy


        dual_bound = 0
        for commodity_index in range(nb_commodities):
            dual_bound += used_dual_var_flow_convexity_per_commoditiy[commodity_index] * convexity_constraint_dict[commodity_index].Rhs

        nb_var_added = 0
        # for each arc, a pricing problem is solved for the pattern variables, if a a pattern has a small enough reduced cost it is added to the formulation
        for arc in arc_list:
            if verbose: print(arc, end='   \r')

            arc_capacity = graph[arc[0]][arc[1]]

            # pricing problem resolution
            new_pattern, subproblem_objective_value = penalized_knapsack_optimizer(demand_list, arc_capacity, used_dual_var_list_per_arc[arc])
            new_pattern_list = [new_pattern]

            dual_bound -= subproblem_objective_value
            reduced_cost = -subproblem_objective_value + used_dual_var_knapsack_convexity_per_arc[arc]

            if reduced_cost < -10**-5: # if the best pattern has a small enough reduced cost it is added to the formulation
                for new_pattern in new_pattern_list:
                    nb_var_added += 1
                    column = gurobipy.Column()
                    column.addTerms(1, knapsack_convexity_constraint_dict[arc])

                    for commodity_index in new_pattern:
                        if capacity_constraint_dict[arc][commodity_index] is not None:
                            column.addTerms(-1, capacity_constraint_dict[arc][commodity_index])

                    pattern_cost = max(0, sum(demand_list[commodity_index] for commodity_index in new_pattern) - arc_capacity)
                    new_var = model.addVar(obj=pattern_cost, column=column)
                    added_var_list.append(new_var)
                    pattern_var_and_cost_per_arc[arc].append((new_pattern, new_var, pattern_cost))

        if verbose : print("Nb added var = ", nb_var_added, ", Nb total var = ", len(added_var_list), ", Dual_bound = ", dual_bound)
        bounds_and_time_list.append((model.ObjVal, dual_bound, time.time() - starting_time))

        if best_dual_bound is None or dual_bound > best_dual_bound:
            best_dual_bound = dual_bound

        if abs(model.ObjVal - best_dual_bound) < 10**-2: # the column generation stops if the bounds are close enough
            break

        if nb_var_added == 0: # the column generation stops if no new variable can be added to the master model after disabaling stabilizations
            if stabilisation == "":
                break
            elif stabilisation == "interior_point" and model.Params.BarConvTol > 10**-3:
                model.Params.BarConvTol = model.Params.BarConvTol / 3
                print("BarConvTol = ", model.Params.BarConvTol)
            else: # stabilisations are disabled in the final iterations of the column generation procedure
                stabilisation = ""
                model.Params.Method = 3
                model.Params.BarConvTol = 10**-8
                model.Params.Crossover = -1

    model.update()
    model.optimize()


def run_DW_Fenchel_model(graph, commodity_list, separation_options=(True, True, True), possible_paths_per_commodity=None, nb_initial_path_created=4, var_delete_proba=0.3,
                            bounds_and_time_list=[], nb_iterations=10**5, verbose=1):
    """
    This algorithm implements the new decomposition method highlighted by this code
    It uses a Dantzig-Wolfe master problem and a Fenchel master problem
    Its subproblem is a Fenchel subproblem with a special normalisation called "directionnal normalisation"
    The value of the computed bounds after convergence is the same as the one computed by a Dantzig-Wolfe decomposition algorithm
    This function also implements other decomposition methods such as the Fenchel decomposition depending on the separation_option chosen

    Inputs:
    graph : graph of the unsplittable flow instance
    commodity_list : commodity list of the unsplittable flow instance
    possible_paths_per_commodity=None : a list of allowed paths for each commodity, if None the 4 k-shortest paths are generated
    separation_options : tuple containing three booleeans : in_out_separation, preprocessing, iterative_separation
    in_out_separation : True the directionnal normalisation is used in the subproblem (DW-Fenchel decompostion), False the natural normalization for the unsplittable flow is used (Fenchel decomposition)
    preprocessing : whether or not preprocessing is used in the Fenchel separation sub-problem
    iterative_separation : can be True only if in_out_separation=True, decide whether the iterative method (see paper) for the resolution of the Fenchel sub-problem is used

    Outputs: None
    """

    nb_nodes = len(graph)
    nb_commodities = len(commodity_list)
    arc_list = [(node, neighbor) for node in range(len(graph)) for neighbor in graph[node]]
    demand_list = [demand for origin, destination, demand in commodity_list]
    starting_time = time.time()

    # creates a set of allowed paths for each commodity, not all paths are consdered allowed in the formulation
    if possible_paths_per_commodity is None:
        possible_paths_per_commodity = compute_possible_paths_per_commodity(graph, commodity_list, nb_initial_path_created)

    # creates the two master problems
    inner_model, inner_variables, inner_constraints = create_knapsack_model(graph, commodity_list, possible_paths_per_commodity, verbose=verbose>1)
    outer_model, outer_variables, outer_constraints = create_arc_path_model(graph, commodity_list, possible_paths_per_commodity, verbose=verbose>1)

    inner_path_and_var_per_commodity, inner_pattern_var_and_cost_per_arc = inner_variables
    outer_path_and_var_per_commodity, outer_overload_vars = outer_variables

    # outer_flow_var_dict[arc][commodity_index] contains the sum of varaibles of the Fenchel master problem that indiquates the flow send by a commodity on an arc
    outer_flow_var_dict = {arc : [0]*nb_commodities for arc in arc_list}
    for commodity_index, path_and_var in enumerate(outer_path_and_var_per_commodity):
        for path, var in path_and_var:
            for node_index in range(len(path)-1):
                arc = (path[node_index], path[node_index+1])
                outer_flow_var_dict[arc][commodity_index] += var

    # inner_flow_var_dict[arc][commodity_index] contains the sum of varaibles of the Dantzig-Wolfe master problem that indiquates the flow send by a commodity on an arc
    inner_flow_var_dict = {arc : [0]*nb_commodities for arc in arc_list}
    for commodity_index, path_and_var in enumerate(inner_path_and_var_per_commodity):
        for path, var in path_and_var:
            for node_index in range(len(path)-1):
                arc = (path[node_index], path[node_index+1])
                inner_flow_var_dict[arc][commodity_index] += var

    # parameters of the Dantzig-Wolfe master model
    inner_model.Params.OutputFlag = 0
    # inner_model.Params.Method = 2
    # inner_model.Params.BarConvTol = 10**-3 # precision of the interior point method
    # inner_model.Params.Crossover = 0

    # parameters of the Fenchel master model
    outer_model.Params.OutputFlag = 0

    # main loop of the algorithm
    for iter_index in range(nb_iterations):
        if verbose : print("iteration : ", iter_index)

        # resolution of the two master models
        inner_model.update()
        inner_model.optimize()
        outer_model.update()
        outer_model.optimize()

        try:
            if verbose : print("Objective function values : DW", inner_model.ObjVal, "F", outer_model.ObjVal)
            if verbose : print("Master model runtimes: DW", inner_model.Runtime, "F", outer_model.Runtime)
            bounds_and_time_list.append((inner_model.ObjVal, outer_model.ObjVal, time.time() - starting_time))

        except:
            print(inner_model.Status)
            print(outer_model.Status)
            import pdb; pdb.set_trace()

        # the method stops if the bounds are close enough
        if abs(inner_model.ObjVal - outer_model.ObjVal) < 10**-3:
            break

        # variable deletion in the Dantzig-Wolfe model to prevent it from becoming to heavy
        for arc in arc_list:
            l = []
            for pattern, var, pattern_cost in inner_pattern_var_and_cost_per_arc[arc]:
                if var.Vbasis != 0 and random.random() < var_delete_proba:
                    inner_model.remove(var)
                else:
                    l.append((pattern, var, pattern_cost))
            inner_pattern_var_and_cost_per_arc[arc] = l

        # subproblem resolution + adding variables and constraints to the two master problems
        nb_separated_arc = apply_fenchel_subproblem(graph, demand_list, outer_model, outer_overload_vars, outer_flow_var_dict,
                                            inner_model, inner_flow_var_dict, inner_pattern_var_and_cost_per_arc, inner_constraints, separation_options,
                                            verbose=verbose)

        if verbose : print("nb_separated_arc = ", nb_separated_arc)


    inner_model.update()
    inner_model.optimize()

    outer_model.update()
    outer_model.optimize()


def apply_fenchel_subproblem(graph, demand_list, outer_model, outer_overload_vars, outer_flow_var_dict,
                                    inner_model, inner_flow_var_dict, inner_pattern_var_and_cost_per_arc, inner_constraints, separation_options, verbose=1):
    # this method calls the algorithms solving a Fenchel like separation subproblem
    # the cuts and variables (here pattern variables) created are added to the two master problems
    arc_list = [(node, neighbor) for node in range(len(graph)) for neighbor in graph[node]]
    nb_commodities = len(demand_list)

    convexity_constraint_dict, knapsack_convexity_constraint_dict, capacity_constraint_dict = inner_constraints

    nb_separated_arc = 0

    t = [0]*5

    for arc in arc_list: # a subproblem is solved for each arc
        temp = time.time()
        arc_capacity = graph[arc[0]][arc[1]]
        outer_flow_vars = outer_flow_var_dict[arc]
        outer_flow_per_commodity = np.array([0 if vars is 0 else vars.getValue() for vars in outer_flow_vars])
        outer_overload_value = outer_overload_vars[arc].X

        inner_flow_vars = inner_flow_var_dict[arc]
        inner_flow_per_commodity = np.array([0 if vars is 0 else vars.getValue() for vars in inner_flow_vars])
        inner_overload_value = sum(var.X * pattern_cost for pattern, var, pattern_cost in inner_pattern_var_and_cost_per_arc[arc])

        t[0] += time.time() - temp
        temp = time.time()

        # call to the separation problem for one arc : a cut and some patterns (vertices of the capacity contraint polyhedron) are returned
        in_out_separation, preprocessing, iterative_separation = separation_options # see definition in run_DW_Fenchel_model
        if in_out_separation:
            if preprocessing:
                constraint_coeff, pattern_cost_and_amount_list = in_out_separation_decomposition_with_preprocessing(demand_list, outer_flow_per_commodity, outer_overload_value, inner_flow_per_commodity, inner_overload_value, arc_capacity, iterative_separation=iterative_separation)
            else:
                constraint_coeff, pattern_cost_and_amount_list = in_out_separation_decomposition(demand_list, outer_flow_per_commodity, outer_overload_value, inner_flow_per_commodity, inner_overload_value, arc_capacity)

        else:
            if preprocessing:
                constraint_coeff, pattern_cost_and_amount_list = separation_decomposition_with_preprocessing(demand_list, outer_flow_per_commodity, arc_capacity)
            else:
                constraint_coeff, pattern_cost_and_amount_list = separation_decomposition(demand_list, outer_flow_per_commodity, arc_capacity)

        commodity_coeff_list, overload_coeff, constant_coeff = constraint_coeff

        t[1] += time.time() - temp
        temp = time.time()

        # if the created cut cuts the solution of the Fenchel master problem it is added to the Fenchel master problem
        if sum(outer_flow_per_commodity * commodity_coeff_list) > constant_coeff + 10**-7 + overload_coeff * outer_overload_value:
            outer_model.addConstr((sum(outer_flow_var * coefficient for outer_flow_var, coefficient in zip(outer_flow_vars, commodity_coeff_list)) - overload_coeff * outer_overload_vars[arc] <= constant_coeff))
            nb_separated_arc += 1

        # the created patterns are added to the Dantzig-Wolfe master problem
        for pattern, pattern_cost, amount in pattern_cost_and_amount_list:
            column = gurobipy.Column()
            column.addTerms(1, knapsack_convexity_constraint_dict[arc])

            for commodity_index in pattern:
                if capacity_constraint_dict[arc][commodity_index] is not None:
                    column.addTerms(-1, capacity_constraint_dict[arc][commodity_index])

            new_var = inner_model.addVar(obj=pattern_cost, column=column)
            inner_pattern_var_and_cost_per_arc[arc].append((pattern, new_var, pattern_cost))

        t[2] += time.time() - temp

    return nb_separated_arc


def compute_possible_paths_per_commodity(graph, commodity_list, nb_initial_path_created):
    # creates a list of allowed paths for each commodity which contains the k-shortest paths for this commodity
    shortest_paths_per_origin = {}
    possible_paths_per_commodity = []

    for commodity_index, commodity in enumerate(commodity_list):
        origin, destination, demand = commodity

        if origin not in shortest_paths_per_origin:
            shortest_paths_per_origin[origin] = k_shortest_path_all_destination(graph, origin, nb_initial_path_created)

        path_and_cost_list = shortest_paths_per_origin[origin][destination]
        possible_paths_per_commodity.append(set(tuple(remove_cycle_from_path(path)) for path, path_cost in path_and_cost_list)) # cycles in the created paths are removed

        possible_paths_per_commodity[commodity_index] = [list(path_tuple) for path_tuple in possible_paths_per_commodity[commodity_index]]

    return possible_paths_per_commodity


def remove_cycle_from_path(path):
    is_in_path = set()
    new_path = []

    for node in path:
        if node in is_in_path:
            while new_path[-1] != node:
                node_to_remove = new_path.pop()
                is_in_path.remove(node_to_remove)

        else:
            is_in_path.add(node)
            new_path.append(node)

    return new_path


if __name__ == "__main__":
    pass
