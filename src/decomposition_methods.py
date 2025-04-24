import numpy as np
import random
import time
import heapq as hp
import gurobipy

from src.k_shortest_path import k_shortest_path_all_destination
from src.models import create_arc_path_model, create_knapsack_model, create_arc_node_model
import src.knapsack_oracles as ko
from plot_results import dico_info


def knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=None, nb_initial_path_created=4, var_delete_proba=0.,
                            flow_penalisation=1, nb_iterations=10**5, gap_limit=10**-4, bounds_and_time_list=[], stabilisation="interior_point", verbose=1, path_generation_loop=False):
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

    if possible_paths_per_commodity is None:
        possible_paths_per_commodity = compute_possible_paths_per_commodity(graph, commodity_list, nb_initial_path_created)

    model, variables, constraints = create_knapsack_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=flow_penalisation, verbose=verbose>1)

    run_knapsack_model(graph, commodity_list, model, variables, constraints, stabilisation, bounds_and_time_list=bounds_and_time_list, nb_iterations=nb_iterations, path_generation_loop=path_generation_loop, var_delete_proba=var_delete_proba, gap_limit=gap_limit, flow_penalisation=flow_penalisation, verbose=verbose)


def run_knapsack_model(graph, commodity_list, model, variables, constraints, stabilisation, bounds_and_time_list=[], nb_iterations=10**5, initial_BarConvTol=10**-1, var_delete_proba=0.0, path_generation_loop=False, flow_penalisation=1, gap_limit=10**-4, verbose=1):
    # column generation process used to solve the linear relaxation of a knapsack model (see create_knapsack_model) of the unsplittable flow problem
    # the knaspsack model is a Dantzig-Wolfe decompostion of the classical model for the unsplittable flow problem
    nb_commodities = len(commodity_list)
    nb_nodes = len(graph)
    demand_list = [commodity[2] for commodity in commodity_list]
    arc_list = [(node, neighbor) for node in range(len(graph)) for neighbor in graph[node]]
    convexity_constraint_dict, knapsack_convexity_constraint_dict, linking_constraint_dict = constraints
    path_and_var_per_commodity, pattern_and_var_per_arc, deviation_variables = variables
    starting_time = time.time()
    nb_var_added = 0
    best_dual_bound = None
    used_dual_var_list_per_arc = None

    # parameter of the solver gurobi
    model.Params.Threads = 1
    model.Params.Method = 2
    model.Params.OutputFlag = 0
    if stabilisation == "interior_point": # in this stabilisation, the master model is solved approximatly (10**-3 precision) with an interior point method
        model.update()
        model.optimize()
        model.Params.Method = 2
        model.Params.BarConvTol = initial_BarConvTol # precision of the interior point method
        model.Params.Crossover = 0

    if stabilisation == "in_out":
        in_out_coeff = 0.9 
        possible_paths_per_commodity = [[path for path, var in path_and_var] for commodity_index, path_and_var in enumerate(path_and_var_per_commodity)]
        arc_path_model, _, arc_path_constraints = create_arc_path_model(graph, commodity_list, possible_paths_per_commodity, verbose=verbose>1)
        arc_path_convexity_constraint_dict, arc_path_capacity_constraint_dict = arc_path_constraints

        arc_path_model.update()
        arc_path_model.optimize()

        in_dual_var_knapsack_convexity_per_arc = {arc : -arc_path_capacity_constraint_dict[arc].RHS * arc_path_capacity_constraint_dict[arc].Pi for arc in arc_list}
        in_dual_var_list_per_arc = {arc : -np.array(demand_list) * arc_path_capacity_constraint_dict[arc].Pi for arc in arc_list}
        in_dual_var_flow_convexity_per_commoditiy = np.array([arc_path_convexity_constraint_dict[commodity_index].Pi for commodity_index in range(nb_commodities)])


    for iter_index in range(nb_iterations):
        if verbose : print("iteration : ", iter_index)

        def solve_model(model):
            model.update()
            model.optimize()
        solve_model(model)

        # getting the dual variables of the master model
        dual_var_knapsack_convexity_per_arc = {arc : -knapsack_convexity_constraint_dict[arc].Pi for arc in arc_list}
        dual_var_flow_convexity_per_commoditiy = np.array([convexity_constraint_dict[commodity_index].Pi for commodity_index in range(nb_commodities)])
        dual_var_list_per_arc = {arc : np.array([-constraint.Pi if constraint is not None else 0 for constraint in linking_constraint_dict[arc]]) for arc in arc_list}

        if stabilisation == "momentum" and used_dual_var_list_per_arc is not None: # another stabilisation, dual variables are aggregated through the iterations in a momentum fashion
            momentum_coeff = 0.8
            used_dual_var_list_per_arc = {arc : momentum_coeff * used_dual_var_list_per_arc[arc] + (1 - momentum_coeff) * dual_var_list_per_arc[arc] for arc in arc_list}
            used_dual_var_knapsack_convexity_per_arc = {arc : momentum_coeff * used_dual_var_knapsack_convexity_per_arc[arc] + (1 - momentum_coeff) * dual_var_knapsack_convexity_per_arc[arc] for arc in arc_list}
            used_dual_var_flow_convexity_per_commoditiy = momentum_coeff * used_dual_var_flow_convexity_per_commoditiy + (1 - momentum_coeff) * dual_var_flow_convexity_per_commoditiy

        elif stabilisation == "in_out" and in_dual_var_list_per_arc != None:
            used_dual_var_list_per_arc = {arc : in_out_coeff * in_dual_var_list_per_arc[arc] + (1 - in_out_coeff) * dual_var_list_per_arc[arc] for arc in arc_list}
            used_dual_var_knapsack_convexity_per_arc = {arc : in_out_coeff * in_dual_var_knapsack_convexity_per_arc[arc] + (1 - in_out_coeff) * dual_var_knapsack_convexity_per_arc[arc] for arc in arc_list}
            used_dual_var_flow_convexity_per_commoditiy = in_out_coeff * in_dual_var_flow_convexity_per_commoditiy + (1 - in_out_coeff) * dual_var_flow_convexity_per_commoditiy

        else:
            used_dual_var_list_per_arc = dual_var_list_per_arc
            used_dual_var_knapsack_convexity_per_arc = dual_var_knapsack_convexity_per_arc
            used_dual_var_flow_convexity_per_commoditiy = dual_var_flow_convexity_per_commoditiy

        dual_bound = 0
        for commodity_index in range(nb_commodities):
            dual_bound += used_dual_var_flow_convexity_per_commoditiy[commodity_index] * convexity_constraint_dict[commodity_index].Rhs

        if path_generation_loop:
            if verbose: print("path generation", end='\r')
            dual_val_grap_per_commodity = [[{neighbor : used_dual_var_list_per_arc[node, neighbor][commodity_index] for neighbor in graph[node]} for node in range(nb_nodes)] for commodity_index in range(nb_commodities)]
            generated_path_list = generate_paths(commodity_list, used_dual_var_flow_convexity_per_commoditiy, dual_val_grap_per_commodity, flow_penalisation=flow_penalisation)
            add_new_paths_to_inner_model(generated_path_list, demand_list, model, path_and_var_per_commodity, convexity_constraint_dict, linking_constraint_dict, flow_penalisation=flow_penalisation)
            dual_bound += sum(path_reduced_cost for _, _, path_reduced_cost in generated_path_list)

        nb_var_added = 0
        new_pattern_list = []
        # for each arc, a pricing problem is solved for the pattern variables, if a a pattern has a small enough reduced cost it is added to the formulation
        for arc in arc_list:
            if verbose: print(arc, end='   \r')

            arc_capacity = graph[arc[0]][arc[1]]
            dual_var_list = used_dual_var_list_per_arc[arc]

            # pricing problem resolution
            if sum(demand_list[commodity_index] for commodity_index in range(nb_commodities) if dual_var_list[commodity_index] != 0) <= arc_capacity:
                new_pattern = [commodity_index for commodity_index, dual_value in enumerate(dual_var_list) if dual_value != 0]
                subproblem_objective_value = sum(dual_var_list)

            else:
                new_pattern, subproblem_objective_value = ko.knapsack_solver(demand_list, arc_capacity, dual_var_list)
            
            dual_bound -= subproblem_objective_value
            reduced_cost = used_dual_var_knapsack_convexity_per_arc[arc] - subproblem_objective_value
            new_pattern_list.append((arc, new_pattern, reduced_cost))

            if reduced_cost < -10**-5: # if the best pattern has a small enough reduced cost it is added to the formulation
                nb_var_added += 1
                column = gurobipy.Column()
                column.addTerms(1, knapsack_convexity_constraint_dict[arc])

                for commodity_index in new_pattern:
                    if linking_constraint_dict[arc][commodity_index] is not None:
                        column.addTerms(-1, linking_constraint_dict[arc][commodity_index])

                new_var = model.addVar(column=column)
                pattern_and_var_per_arc[arc].append((new_pattern, new_var))

        if best_dual_bound is None or dual_bound > best_dual_bound:
            best_dual_bound = dual_bound
        gap = (model.ObjVal - best_dual_bound) / (abs(best_dual_bound) + 10**-8)
        if stabilisation == "interior_point" and model.Params.BarConvTol > gap/2:
            model.Params.BarConvTol = max(0, gap/2)
            if verbose : print("BarConvTol = ", model.Params.BarConvTol)

        if verbose : print("Objective bound : primal", model.ObjVal, "dual", dual_bound, "gap", gap)
        if verbose : print("Master model runtime : ", model.Runtime)
        nb_pattern_vars = sum(len(pattern_and_var_per_arc[arc]) for arc in pattern_and_var_per_arc)
        nb_path_vars = sum(len(path_and_var) for path_and_var in path_and_var_per_commodity)
        if verbose : print("Nb added var = ", nb_var_added, ", Nb total pattern var = ", nb_pattern_vars, ", Nb total path var = ", nb_path_vars)
        bounds_and_time_list.append((model.ObjVal, dual_bound, time.time() - starting_time))


        if gap < gap_limit: # the column generation stops if the bounds are close enough
            break

        if nb_var_added == 0: # the column generation stops if no new variable can be added to the master model after disabaling stabilizations
            if stabilisation == "":
                break
            elif stabilisation == "in_out":
                if 0 == in_out_coeff:
                    break
                in_out_coeff = max(in_out_coeff - 0.2, 0)

                in_dual_var_knapsack_convexity_per_arc = used_dual_var_knapsack_convexity_per_arc
                in_dual_var_list_per_arc = used_dual_var_list_per_arc
                in_dual_var_flow_convexity_per_commoditiy = used_dual_var_flow_convexity_per_commoditiy

            elif stabilisation == "interior_point" and model.Params.BarConvTol > 10**-3:
                model.Params.BarConvTol = model.Params.BarConvTol / 3
                print("BarConvTol = ", model.Params.BarConvTol)
            else: # stabilisations are disabled in the final iterations of the column generation procedure
                stabilisation = ""
                model.Params.Method = -1
                model.Params.BarConvTol = 10**-8
                model.Params.Crossover = -1
        
        elif stabilisation == "in_out":

            subgradient = {arc : np.zeros(nb_commodities) for arc in arc_list}
            for arc, new_pattern, pattern_reduced_cost in new_pattern_list:
                for commodity_index in new_pattern:
                    subgradient[arc][commodity_index] -= 1
            
            if path_generation_loop:
                for commodity_index, path, path_reduced_cost in generated_path_list:
                    for node_index in range(len(path)-1):
                        arc = (path[node_index], path[node_index+1])
                        subgradient[arc][commodity_index] += 1
            else:
                for commodity_index, path_and_var in enumerate(path_and_var_per_commodity):
                    for path, var in path_and_var:
                        for node_index in range(len(path)-1):
                            arc = (path[node_index], path[node_index+1])
                            subgradient[arc][commodity_index] += var.X

            in_out_direction =  {arc : in_dual_var_list_per_arc[arc] - dual_var_list_per_arc[arc] for arc in arc_list}              
            scalar_product = sum(subgradient[arc] @ in_out_direction[arc] for arc in arc_list)
            if scalar_product > 0:
                in_out_coeff = 0.1+ 0.9 * in_out_coeff
            else:
                in_out_coeff = max(0, in_out_coeff - 0.1)


    model.update()
    model.optimize()


def run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=None, nb_initial_path_created=4, separation_options=(True, True, True), var_delete_proba=0.3,
                            bounds_and_time_list=[], nb_iterations=10**5, path_generation_loop=False, flow_penalisation=1, gap_limit=10**-4, verbose=1):
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

    nb_commodities = len(commodity_list)
    nb_nodes = len(graph)
    arc_list = [(node, neighbor) for node in range(len(graph)) for neighbor in graph[node]]
    demand_list = [demand for origin, destination, demand in commodity_list]
    starting_time = time.time()
    ko.nb_calls = 0

    # creates a set of allowed paths for each commodity, not all paths are consdered allowed in the formulation
    if possible_paths_per_commodity is None:
        possible_paths_per_commodity = compute_possible_paths_per_commodity(graph, commodity_list, nb_initial_path_created)

    # creates the two master problems
    inner_model, inner_variables, inner_constraints = create_knapsack_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=flow_penalisation, verbose=verbose>1)
    outer_model, outer_variables, outer_constraints = create_arc_path_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=flow_penalisation, verbose=verbose>1)

    inner_path_and_var_per_commodity, inner_pattern_and_var_per_arc, inner_deviation_variables = inner_variables
    outer_path_and_var_per_commodity, outer_deviation_vars = outer_variables
    outer_convexity_constraint_dict, capacity_constraint_dict = outer_constraints
    inner_convexity_constraint_dict, knapsack_convexity_constraint_dict, linking_constraint_dict = inner_constraints

    for arc in arc_list:
        capacity_constraint_dict[arc] = [(capacity_constraint_dict[arc], demand_list, graph[arc[0]][arc[1]])]


    # parameters of the Dantzig-Wolfe master model
    inner_model.Params.OutputFlag = 0
    inner_model.Params.Threads = 1

    # inner_model.Params.Method = 2
    # inner_model.Params.BarConvTol = 10**-5 # precision of the interior point method
    # inner_model.Params.Crossover = 0

    # parameters of the Fenchel master model
    outer_model.Params.OutputFlag = 0
    outer_model.Params.Threads = 1

    # main loop of the algorithm
    for iter_index in range(nb_iterations):
        if verbose : print("iteration : ", iter_index)

        # resolution of the two master models
        def solve_model(model):
            model.update()
            model.optimize()
        solve_model(outer_model)

        # inner_model.Params.Crossover = 0
        # solve_model(inner_model)
        # dual_var_list_per_arc = {arc : np.array([-constraint.Pi if constraint is not None else 0 for constraint in linking_constraint_dict[arc]]) for arc in arc_list}
        # inner_model.Params.Crossover = 1
        solve_model(inner_model)


        primal_bound = inner_model.ObjVal
        dual_bound = outer_model.ObjVal

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

        if path_generation_loop:
            if verbose: print("path gen", end='\r')
            # Computing dual values for path generation
            inner_dual_var_graph_per_commodity = [[{neighbor : -linking_constraint_dict[node, neighbor][commodity_index].Pi + 10**-5 if linking_constraint_dict[node, neighbor][commodity_index] is not None else 0 for neighbor in graph[node]} for node in range(nb_nodes)] for commodity_index in range(nb_commodities)]
            inner_convexity_dual_var_list = [inner_convexity_constraint_dict[commodity_index].Pi for commodity_index in range(nb_commodities)]
            outer_dual_var_graph_per_commodity = [[{neighbor : -sum(constraint.Pi * coeff_list[commodity_index] for constraint, coeff_list, constant_coeff in capacity_constraint_dict[node, neighbor]) + 10**-5 for neighbor in graph[node]} for node in range(nb_nodes)] for commodity_index in range(nb_commodities)]
            outer_convexity_dual_var_list = [outer_convexity_constraint_dict[commodity_index].Pi for commodity_index in range(nb_commodities)]

            generated_path_list_from_inner_model = generate_paths(commodity_list, inner_convexity_dual_var_list, inner_dual_var_graph_per_commodity, flow_penalisation=flow_penalisation)
            generated_path_list_from_outer_model = generate_paths(commodity_list, outer_convexity_dual_var_list, outer_dual_var_graph_per_commodity, flow_penalisation=flow_penalisation)

            print(sum(path_reduced_cost for _, _, path_reduced_cost in generated_path_list_from_outer_model if path_reduced_cost <0))
            dual_bound += sum(path_reduced_cost for _, _, path_reduced_cost in generated_path_list_from_outer_model)
            new_path_list = generated_path_list_from_inner_model + generated_path_list_from_outer_model

        gap = (primal_bound - dual_bound) / (abs(dual_bound) + 10**-8)
        try:
            if verbose : print("Objective bounds : primal", primal_bound, "dual", dual_bound, "gap", gap)
            if verbose : print("Master model runtimes: DW", inner_model.Runtime, "F", outer_model.Runtime)
            bounds_and_time_list.append((primal_bound, dual_bound, time.time() - starting_time))

        except:
            print(inner_model.Status)
            print(outer_model.Status)
            import pdb; pdb.set_trace()

        # the method stops if the bounds are close enough
        if gap < gap_limit:
            break

        # # variable deletion in the Dantzig-Wolfe model to prevent it from becoming to heavy
        # for arc in arc_list:
        #     l = []
        #     for pattern, var in inner_pattern_and_var_per_arc[arc]:
        #         if var.Vbasis != 0 and random.random() < var_delete_proba:
        #             inner_model.remove(var)
        #         else:
        #             l.append((pattern, var))
        #     inner_pattern_and_var_per_arc[arc] = l

        temp = time.time()
        # subproblem resolution + adding variables and constraints to the two master problems
        nb_separated_arc = apply_fenchel_subproblem(graph, demand_list, outer_model, outer_flow_var_dict,
                                            inner_model, inner_flow_var_dict, inner_pattern_and_var_per_arc, inner_constraints, outer_constraints, separation_options,
                                            verbose=verbose)
        
        if not "nb_separated_arc" in dico_info: dico_info["nb_separated_arc"] = []
        dico_info["nb_separated_arc"].append(nb_separated_arc)

        if verbose : print("Subproblem time = ", time.time() - temp)

        if path_generation_loop:
            add_new_paths_to_inner_model(new_path_list, demand_list, inner_model, inner_path_and_var_per_commodity, inner_convexity_constraint_dict, linking_constraint_dict, flow_penalisation=flow_penalisation)
            add_new_paths_to_outer_model(new_path_list, demand_list, outer_model, outer_path_and_var_per_commodity, outer_convexity_constraint_dict, capacity_constraint_dict, flow_penalisation=flow_penalisation)
            

        if verbose : print("nb_separated_arc = ", nb_separated_arc)
        if verbose : print("mean_nb_calls = ", ko.nb_calls / (iter_index+1) / len(arc_list))


    inner_model.update()
    inner_model.optimize()

    outer_model.update()
    outer_model.optimize()

    return {arc : [(coeff_list, constant_coeff) for constraint, coeff_list, constant_coeff in capacity_constraint_dict[arc]] for arc in arc_list}


def apply_fenchel_subproblem(graph, demand_list, outer_model, outer_flow_var_dict,
                                    inner_model, inner_flow_var_dict, inner_pattern_and_var_per_arc, inner_constraints, outer_constraints, separation_options, dual_var_list_per_arc=None, verbose=1):
    # this method calls the algorithms solving a Fenchel like separation subproblem
    # the cuts and variables (here pattern variables) created are added to the two master problems
    arc_list = [(node, neighbor) for node in range(len(graph)) for neighbor in graph[node]]
    nb_commodities = len(demand_list)

    convexity_constraint_dict, knapsack_convexity_constraint_dict, linking_constraint_dict = inner_constraints


    nb_separated_arc = 0

    t = [0]*5


    for arc in arc_list: # a subproblem is solved for each arc
        temp = time.time()
        arc_capacity = graph[arc[0]][arc[1]]
        outer_flow_per_commodity = np.array([0 if vars is 0 else vars.getValue() for vars in outer_flow_var_dict[arc]])

        inner_flow_per_commodity = np.array([0 if vars is 0 else vars.getValue() for vars in inner_flow_var_dict[arc]])

        t[0] += time.time() - temp
        temp = time.time()

        # call to the separation problem for one arc : a cut and some patterns (vertices of the capacity contraint polyhedron) are returned
        in_out_separation, preprocessing, iterative_separation = separation_options # see definition in run_DW_Fenchel_model
        if in_out_separation:
            if preprocessing:
                constraint_coeff, pattern_and_amount_list = ko.in_out_separation_decomposition_with_preprocessing(demand_list, outer_flow_per_commodity, inner_flow_per_commodity, arc_capacity, iterative_separation=iterative_separation)
            else:
                constraint_coeff, pattern_and_amount_list = ko.in_out_separation_decomposition(demand_list, outer_flow_per_commodity, inner_flow_per_commodity, arc_capacity)

        else:
            if preprocessing:
                constraint_coeff, pattern_and_amount_list = ko.separation_decomposition_with_preprocessing(demand_list, outer_flow_per_commodity, arc_capacity)
            else:
                constraint_coeff, pattern_and_amount_list = ko.separation_decomposition(demand_list, outer_flow_per_commodity, arc_capacity)

        commodity_coeff_list, constant_coeff = constraint_coeff

        t[1] += time.time() - temp
        temp = time.time()

        # if the created cut cuts the solution of the Fenchel master problem it is added to the Fenchel master problem
        if sum(outer_flow_per_commodity * commodity_coeff_list) > constant_coeff + 10**-5:
            new_constraint = outer_model.addConstr((sum(outer_flow_var * coefficient for outer_flow_var, coefficient in zip(outer_flow_var_dict[arc], commodity_coeff_list)) <= constant_coeff))
            nb_separated_arc += 1
            outer_constraints[1][arc].append((new_constraint, commodity_coeff_list, constant_coeff))

        # if dual_var_list_per_arc is not None:
        #     dual_vars = dual_var_list_per_arc[arc]
        #     dual_vars = np.floor(dual_vars * 10**4)
        #     new_pattern, subproblem_objective_value = ko.knapsack_solver(demand_list, arc_capacity, dual_var_list_per_arc[arc])
        #     for pattern, _ in inner_pattern_and_var_per_arc[arc] + pattern_and_amount_list:
        #         if pattern == new_pattern:
        #             break
        #     else:
        #         pattern_and_amount_list.append((new_pattern, 0))

        #     if sum(outer_flow_per_commodity * dual_var_list_per_arc[arc]) > subproblem_objective_value + 10**-5:
        #         print("ez")
        #         new_constraint = outer_model.addConstr((sum(outer_flow_var * coefficient for outer_flow_var, coefficient in zip(outer_flow_var_dict[arc], dual_var_list_per_arc[arc])) <= subproblem_objective_value))
        #         outer_constraints[1][arc].append((new_constraint, dual_var_list_per_arc[arc], subproblem_objective_value))

        # the created patterns are added to the Dantzig-Wolfe master problem
        for pattern, amount in pattern_and_amount_list:
            column = gurobipy.Column()
            column.addTerms(1, knapsack_convexity_constraint_dict[arc])

            for commodity_index in pattern:
                if linking_constraint_dict[arc][commodity_index] is not None:
                    column.addTerms(-1, linking_constraint_dict[arc][commodity_index])
            
            new_var = inner_model.addVar(column=column)
            inner_pattern_and_var_per_arc[arc].append((pattern, new_var))

            if not "nb_added_points" in dico_info: dico_info["nb_added_points"] = 0
            dico_info["nb_added_points"] += 1

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


def generate_paths(commodity_list, convexity_dual_var_list, dual_var_graph_per_commodity, flow_penalisation=1):
    nb_nodes = len(dual_var_graph_per_commodity[0])
    nb_commodities = len(commodity_list)
    max_demand = max(demand for origin, destination, demand in commodity_list)

    generated_path_list = []

    for commodity_index in range(nb_commodities):
        origin, destination, demand = commodity_list[commodity_index]
        graph = [{neighbor : dual_var_dict[neighbor] + flow_penalisation  for neighbor in dual_var_dict} for node, dual_var_dict in enumerate(dual_var_graph_per_commodity[commodity_index])]
        
        shortest_path, path_cost = dijkstra(graph, origin, destination)

        reduced_cost = path_cost - convexity_dual_var_list[commodity_index]
        generated_path_list.append((commodity_index, shortest_path, reduced_cost))
    
    return generated_path_list


def add_new_paths_to_inner_model(new_path_list, demand_list, model, path_and_var_per_commodity, convexity_constraint_dict, linking_constraint_dict, flow_penalisation=1):
    max_demand = max(demand_list)
    for commodity_index, path, path_reduced_cost in new_path_list:
        demand = demand_list[commodity_index]
        if path_reduced_cost < -10**-3:
            column = gurobipy.Column()
            column.addTerms(1, convexity_constraint_dict[commodity_index])

            for node_index in range(len(path)-1):
                node, neighbor = path[node_index], path[node_index+1]

                if linking_constraint_dict[node, neighbor][commodity_index] is None:
                    linking_constraint_dict[node, neighbor][commodity_index] = model.addConstr((gurobipy.LinExpr(0) <= 0), "capacity")

                column.addTerms(1, linking_constraint_dict[node, neighbor][commodity_index])

            new_var = model.addVar(obj=flow_penalisation * (len(path)-1) , column=column)
            path_and_var_per_commodity[commodity_index].append((path, new_var))



def add_new_paths_to_outer_model(new_path_list, demand_list, model, path_and_var_per_commodity, convexity_constraint_dict, capacity_constraint_dict, flow_penalisation=1):
    max_demand = max(demand_list)
    for commodity_index, path, path_reduced_cost in new_path_list:
        demand = demand_list[commodity_index]
        if path_reduced_cost < -10**-3:
            column = gurobipy.Column()
            column.addTerms(1, convexity_constraint_dict[commodity_index])

            for node_index in range(len(path)-1):
                node, neighbor = path[node_index], path[node_index+1]

                for constraint, coeff_list, constant_coeff in capacity_constraint_dict[node, neighbor]:
                    column.addTerms(coeff_list[commodity_index], constraint)

            new_var = model.addVar(obj=flow_penalisation * (len(path)-1) , column=column)
            path_and_var_per_commodity[commodity_index].append((path, new_var))
        


def dijkstra(graph, intial_node, destination_node=None):
    priority_q = [(0, intial_node, None)]
    parent_list = [None] * len(graph)
    distances = [None] * len(graph)

    while priority_q:
        value, current_node, parent_node = hp.heappop(priority_q)

        if distances[current_node] is None:
            parent_list[current_node] = parent_node
            distances[current_node] = value

            if current_node == destination_node:
                break

            for neighbor in graph[current_node]:
                if distances[neighbor] is None:
                    hp.heappush(priority_q, (value + graph[current_node][neighbor], neighbor, current_node))

    current_node = destination_node
    path = []
    while current_node is not None:
        path.append(current_node)
        current_node = parent_list[current_node]
    path.reverse()

    return path, distances[destination_node]


def gurobi_with_cuts(graph, commodity_list, possible_paths_per_commodity=None, verbose=1):
    arc_list = [(node, neighbor) for node in range(len(graph)) for neighbor in graph[node]]
    nb_commodities = len(commodity_list)

    model, variables, constraints = create_arc_node_model(graph, commodity_list, flow_penalisation=1, verbose=0)
    
    deviation_variables, flow_variables = variables
    capacity_constraint_dict, flow_constraint_dict = constraints

    for commodity_index, var_dic in enumerate(flow_variables):
        for arc in var_dic:
            var_dic[arc].VType = 'B'

    model.Params.OutputFlag = 1
    model.Params.MIPFocus = 3
    model.update()
    model.optimize()


    additional_constraint_dict = run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, nb_iterations=80, separation_options=(True, True, True), path_generation_loop=True, gap_limit=2*10**-3, verbose=verbose)

    for arc in arc_list:
        for coeff_list, constant_coeff in additional_constraint_dict[arc]:
            constraint = model.addConstr(sum(flow_variables[commodity_index][arc] * coeff_list[commodity_index] for commodity_index in range(nb_commodities)) <= constant_coeff)
            constraint.Lazy = -1

    model.reset(1)
    model.update()
    model.optimize()




if __name__ == "__main__":
    pass
