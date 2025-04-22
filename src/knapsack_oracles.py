import random
import numpy as np
import time
import gurobipy

import src.knapsacksolver as knapsacksolver # the code must be able to import knapsacksolver.so which is the result of the compilation of the library made by fontanf : https://github.com/fontanf/knapsacksolver.git, place knapsacksolver.so in the main folder
from plot_results import dico_info


def compute_all_lifted_coefficients(demand_list, variable_pattern, coeff_list, fixed_pattern, RHS, remaining_arc_capacity):
    # this function take a cut valid for a polyhedron on a lower dimension and "lifts" it to become a cut valid for a polyhedron on a higher dimension
    # This is done with a technique called sequential lifting
     # for more details, see the concept of cut lifting (can be found on the literature on knapsack or Fencel cuts)
    lifted_demand_list = [demand_list[commodity_index] for commodity_index in variable_pattern]
    lifted_commodity_list = list(variable_pattern)
    commodity_to_lift_list = list(fixed_pattern)
    coeff_list = list(coeff_list)
    new_pattern_list = []

    while commodity_to_lift_list: # the coefficient of one variable is lifted at every iteration
        commodity_index = commodity_to_lift_list.pop(0)
        remaining_arc_capacity += demand_list[commodity_index]

        pre_pattern, lifted_coeff_part = knapsack_solver(lifted_demand_list, remaining_arc_capacity, coeff_list)
        if not "knapsack_lifting" in dico_info: dico_info["knapsack_lifting"] = 0
        dico_info["knapsack_lifting"] += 1

        pattern = [lifted_commodity_list[index] for index in pre_pattern] + commodity_to_lift_list
        new_pattern_list.append(pattern)

        RHS, lifted_coeff = lifted_coeff_part, lifted_coeff_part - RHS

        lifted_demand_list.append(demand_list[commodity_index])
        lifted_commodity_list.append(commodity_index)
        coeff_list.append(lifted_coeff)

    commodity_all_coeffs = np.zeros(len(demand_list))
    for index, commodity_index in enumerate(variable_pattern + fixed_pattern):
        commodity_all_coeffs[commodity_index] = coeff_list[index]

    return commodity_all_coeffs, RHS, new_pattern_list



def compute_approximate_decomposition(demand_list, flow_per_commodity, arc_capacity, overload_penalization=1, order_of_commodities="sorted"):
    # compute a decomposition of a flow distribution on an arc as a convex combination of commodity patterns.
    # The decompostion is computed in a greedy way to minimize the cost of the patterns used
    # The algorithm start with a (very costly) pattern containing all commodities
    # At each iteration the highest cost pattern is selected and a commodity that is over-represented in the decomposition is removed from it

    nb_commodities = len(demand_list)
    cost_pattern_and_amount_list = [[overload_penalization * max(0, sum(demand_list) - arc_capacity), list(range(nb_commodities)), 1]]

    commodity_order = list(range(nb_commodities))
    if order_of_commodities == "sorted":
        commodity_order.sort(key=lambda x:demand_list[x], reverse=True)
    elif order_of_commodities == "random":
        random.shuffle(commodity_order)

    for commodity_index in commodity_order:
        current_flow = 1
        while current_flow > flow_per_commodity[commodity_index] + 10**-5:
            cost_pattern_and_amount = max([x for x in cost_pattern_and_amount_list if commodity_index in x[1]])
            pattern_overload, pattern, amount = cost_pattern_and_amount
            new_pattern = list(pattern)
            new_pattern.remove(commodity_index)
            new_pattern_overload = max(0, pattern_overload - overload_penalization * demand_list[commodity_index])
            new_amount = min(amount, current_flow - flow_per_commodity[commodity_index])
            cost_pattern_and_amount_list.append([new_pattern_overload, new_pattern, new_amount])
            current_flow -= new_amount

            if new_amount == amount:
                cost_pattern_and_amount_list.remove(cost_pattern_and_amount)
            else:
                cost_pattern_and_amount[2] -= new_amount

    return cost_pattern_and_amount_list


def separation_decomposition(demand_list, flow_per_commodity, arc_capacity, verbose=0):
    # compute a decomposition of a flow distribution on an arc as a convex combination of commodity patterns.
    # the primal variables of the last iteration indiquate the patterns used in the decomposition and the dual variables of the last iteration represent the coefficients of a cut
    # a colum generation process is used to solve the decomposition problem (see explanations on the subproblem of the Fenchel decomposotion in the paper for more details, the decomposition problem is the dual of the separation Fenchel subproblem)
    nb_commodities = len(demand_list)

    # prevent floating point errors
    flow_per_commodity = np.clip(flow_per_commodity, 0, 1)
    flow_per_commodity = 10**-6 * np.floor(flow_per_commodity * 10**6)

    # Create optimization model
    model = gurobipy.Model()
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = 0
    model.Params.Threads = 1

    # starts with an approximate decomposition
    # pattern_and_amount_list = []
    cost_pattern_and_amount_list = compute_approximate_decomposition(demand_list, flow_per_commodity, arc_capacity)

    # Create pattern variables
    pattern_and_var_list = [(pattern, model.addVar()) for pattern_overload, pattern, amount in cost_pattern_and_amount_list if pattern_overload == 0] # pattern choice variables
    penalization_var = model.addVar(obj=1)

    convexity_constraint = model.addConstr(gurobipy.LinExpr([(1, var) for pattern, var in pattern_and_var_list]) == 1)

    knapsack_constraint_dict = {}
    for commodity_index in range(nb_commodities):
        flow_var = gurobipy.LinExpr([(1, var) for pattern, var in pattern_and_var_list if commodity_index in pattern])
        knapsack_constraint_dict[commodity_index] = model.addConstr((penalization_var + flow_var >= flow_per_commodity[commodity_index]))

    # main loop of the column generation
    use_heuristic = True
    i = 0
    while True:
        i += 1
        model.update()
        model.optimize()

        # extracting dual values from the model
        commodity_dual_value_list = np.array([-knapsack_constraint_dict[commodity_index].Pi for commodity_index in range(nb_commodities)])
        convexity_dual_value = convexity_constraint.Pi

        # resolution of the subproblem of the column generation process (heuristic resolution may be used to speed up the process)
        if sum(demand for demand, dual_value in zip(demand_list, commodity_dual_value_list) if dual_value != 0) <= arc_capacity:
            pattern = [commodity_index for commodity_index, dual_value in enumerate(commodity_dual_value_list) if dual_value != 0]
            subproblem_objective_value = -sum(commodity_dual_value_list)

        elif use_heuristic:
            pattern, subproblem_objective_value = approximate_knapsack_optimizer(demand_list, arc_capacity, -commodity_dual_value_list)

        else:
            pattern, subproblem_objective_value = knapsack_solver(demand_list, arc_capacity, -commodity_dual_value_list)

        reduced_cost = -subproblem_objective_value - convexity_dual_value
        if verbose:
            print(i, model.ObjVal, len(demand_list), convexity_dual_value, end='        \r')

        # if a pattern with a negative reduced cost has been computed, it is added to the model
        if reduced_cost < -10**-4:
            use_heuristic = True
            column = gurobipy.Column()
            column.addTerms(1, convexity_constraint)

            for commodity_index in pattern:
                column.addTerms(1, knapsack_constraint_dict[commodity_index])

            new_var = model.addVar(column=column)
            pattern_and_var_list.append((pattern, new_var))

        else:
            if use_heuristic:
                use_heuristic = False
            elif model.Params.Method == 2:
                model.Params.Method = -1
            else:
                break

    return (-commodity_dual_value_list, -convexity_dual_value), [(pattern, var.X) for pattern, var in pattern_and_var_list if var.Vbasis == 0]


def separation_decomposition_with_preprocessing(demand_list, flow_per_commodity, arc_capacity, verbose=0):
    # makes some preprocessing then calls separation_decomposition to make the decomposition and compute the cut
    # afterwards the coefficients of the cuts are lifted
    # for preprocessing details and lifting (see the paper)
    nb_commodities = len(demand_list)

    fixed_pattern = [commodity_index for commodity_index, flow_value in enumerate(flow_per_commodity) if flow_value == 1]
    variable_pattern = [commodity_index for commodity_index, flow_value in enumerate(flow_per_commodity) if flow_value != 1 and flow_value != 0]

    variable_demand_list = [demand_list[commodity_index] for commodity_index in variable_pattern]
    remaining_arc_capacity = arc_capacity - sum(demand_list[commodity_index] for commodity_index in fixed_pattern)
    variable_flow_per_commodity = [flow_per_commodity[commodity_index] for commodity_index in variable_pattern]

    # calling the separation/decomposition method
    constraint_coeff, pre_pattern_and_amount_list = separation_decomposition(variable_demand_list, variable_flow_per_commodity, remaining_arc_capacity, verbose=verbose)

    variable_commodity_coeff_list, constant_coeff = constraint_coeff
    pattern_and_amount_list = [([variable_pattern[index] for index in pattern] + fixed_pattern, amount) for pattern, amount in pre_pattern_and_amount_list]

    # lifting the coefficients of the cut
    commodity_coeff_list, constant_coeff, lifting_pattern_list = compute_all_lifted_coefficients(demand_list, variable_pattern, variable_commodity_coeff_list, fixed_pattern, constant_coeff, remaining_arc_capacity)

    for pattern in lifting_pattern_list:
        pattern_and_amount_list.append((pattern, 0))

    return (commodity_coeff_list, constant_coeff), pattern_and_amount_list


accumulated_time_in_separation_lp = 0
accumulated_time_in_separation_other = 0
def in_out_separation_decomposition(demand_list, outter_flow_per_commodity, inner_flow_per_commodity, arc_capacity, verbose=0):
    # compute a decomposition of a flow distribution on an arc as a convex combination of commodity patterns. This decompostion is optimal in the sense of the directional nomalization
    # the primal variables of the last iteration indiquate the patterns used in the decomposition and the dual variables of the last iteration represent the coefficients of a cut
    # a colum generation process is used to solve the decomposition problem (see explanations on the subproblem of the Fenchel decomposotion in the paper for more details, the decomposition problem is the dual of the separation Fenchel subproblem)
    nb_commodities = len(demand_list)

    # prevents rounding errors
    inner_flow_per_commodity = np.clip(inner_flow_per_commodity, 0, 1)
    inner_flow_per_commodity = 10**-6 * np.floor(inner_flow_per_commodity * 10**6)
    outter_flow_per_commodity = np.clip(outter_flow_per_commodity, 0, 1)
    outter_flow_per_commodity = 10**-6 * np.floor(outter_flow_per_commodity * 10**6)

    # Create optimization model
    model = gurobipy.Model()
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = 0
    model.Params.Threads = 1

    # Create variables
    initial_pattern_var = model.addVar(obj=0)
    pattern_and_var_list = [([], initial_pattern_var)] # pattern choice variables
    penalisation_var_plus = model.addVar(obj=1) # positive part of the penalisation var
    penalisation_var_minus = model.addVar(obj=1) # negative part of the penalisation var
    penalisation_var = penalisation_var_plus - penalisation_var_minus


    cost_pattern_and_amount_list = compute_approximate_decomposition(demand_list, inner_flow_per_commodity, arc_capacity) + compute_approximate_decomposition(demand_list, outter_flow_per_commodity, arc_capacity)
    for pattern_overload, pattern, _ in cost_pattern_and_amount_list:
        if pattern_overload == 0:
            pattern_and_var_list.append((pattern, model.addVar()))

    convexity_constraint = model.addConstr(sum(var for pattern, var in pattern_and_var_list) == 1)

    knapsack_constraint_dict = {}
    for commodity_index in range(nb_commodities):
        inner_flow, outter_flow = inner_flow_per_commodity[commodity_index], outter_flow_per_commodity[commodity_index]
        knapsack_constraint_dict[commodity_index] = model.addConstr(-initial_pattern_var * inner_flow + penalisation_var * (inner_flow - outter_flow) <= -outter_flow)

    # main loop of the column generation process
    i = 0
    while True:
        i += 1
        model.update()
        model.optimize()


        # getting the dual variables
        commodity_dual_value_list = np.array([-knapsack_constraint_dict[commodity_index].Pi for commodity_index in range(nb_commodities)])
        convexity_dual_value = -convexity_constraint.Pi

        # solving the subproblem of the column generation process
        pattern, subproblem_objective_value = knapsack_solver(demand_list, arc_capacity, commodity_dual_value_list)
        if not "knapsack_separation" in dico_info: dico_info["knapsack_separation"] = 0
        dico_info["knapsack_separation"] += 1

        reduced_cost = -subproblem_objective_value + convexity_dual_value
        if verbose : print(i, model.ObjVal, reduced_cost, end='          \r')

        #  if the pattern with a negative reduced cost is computed it is added to the model
        if reduced_cost < -10**-5:
            column = gurobipy.Column()
            column.addTerms(1, convexity_constraint)

            for commodity_index in pattern:
                column.addTerms(-1, knapsack_constraint_dict[commodity_index])

            new_var = model.addVar(obj=0, column=column)
            pattern_and_var_list.append((pattern, new_var))

        else:
            break


    return (commodity_dual_value_list, convexity_dual_value), [(pattern, var.X) for pattern, var in pattern_and_var_list[1:] if var.VBasis == 0]


def in_out_separation_decomposition_iterative(demand_list, outter_flow_per_commodity, inner_flow_per_commodity, arc_capacity):
    # compute a decomposition of a flow distribution on an arc as a convex combination of commodity patterns. This decompostion is optimal in the sense of the directional normalisation
    # the primal variables of the last iteration indiquate the patterns used in the decomposition and the dual variables of the last iteration represent the coefficients of a cut
    # this is the new iterative method to solve the Fenchel subproblem presented in the paper
    # separation/decomposition using another normalisation called repeatedly to make the computation
    nb_commodities = len(demand_list)
    in_out_convex_coeff = 0
    demand_list = np.array(demand_list)
    # inner_flow_per_commodity = np.array(inner_flow_per_commodity)
    # outter_flow_per_commodity = np.array(outter_flow_per_commodity)
    inner_flow_per_commodity = np.clip(inner_flow_per_commodity, 0, 1)
    inner_flow_per_commodity = 10**-6 * np.floor(inner_flow_per_commodity * 10**6)
    outter_flow_per_commodity = np.clip(outter_flow_per_commodity, 0, 1)
    outter_flow_per_commodity = 10**-6 * np.floor(outter_flow_per_commodity * 10**6)
    current_flow_per_commodity = outter_flow_per_commodity
    old_constraint_coeff = (np.zeros(nb_commodities), 0)

    i = 0
    while True:
        constraint_coeff, pattern_and_amount_list = separation_decomposition_with_preprocessing(demand_list, current_flow_per_commodity, arc_capacity, verbose=0)
        commodity_coeff_list, constant_coeff = constraint_coeff

        if sum(commodity_coeff_list * current_flow_per_commodity) <= constant_coeff + 10**-5:
            break
        i+=1

        outter_value = sum(commodity_coeff_list * outter_flow_per_commodity) - constant_coeff
        inner_value = sum(commodity_coeff_list * inner_flow_per_commodity) - constant_coeff
        in_out_convex_coeff = max(0, min(1, (- outter_value) / (inner_value - outter_value)))
        current_flow_per_commodity = in_out_convex_coeff * inner_flow_per_commodity + (1 - in_out_convex_coeff) * outter_flow_per_commodity
        old_constraint_coeff = constraint_coeff


    if not "nb_cuts_iterative" in dico_info: dico_info["nb_cuts_iterative"] = []
    dico_info["nb_cuts_iterative"].append(i)
    return old_constraint_coeff, pattern_and_amount_list


def in_out_separation_decomposition_with_preprocessing(demand_list, outter_flow_per_commodity, inner_flow_per_commodity,
                                                        arc_capacity, iterative_separation=False):
    # makes some preprocessing then calls the method that will make the decomposition and compute the cut
    # afterwards the coefficients of the cuts are lifted
    # for preprocessing details and lifting (see the paper)
    nb_commodities = len(demand_list)
    fixed_pattern = []
    variable_pattern = []

    for commodity_index in range(nb_commodities):
        outter_flow, inner_flow = outter_flow_per_commodity[commodity_index], inner_flow_per_commodity[commodity_index]

        if outter_flow == 0 and inner_flow == 0:
            pass

        elif outter_flow == 1 and inner_flow == 1:
            fixed_pattern.append(commodity_index)

        else:
            variable_pattern.append(commodity_index)

    variable_demand_list = [demand_list[commodity_index] for commodity_index in variable_pattern]
    variable_outter_flow_per_commodity = [outter_flow_per_commodity[commodity_index] for commodity_index in variable_pattern]
    variable_inner_flow_per_commodity = [inner_flow_per_commodity[commodity_index] for commodity_index in variable_pattern]
    remaining_arc_capacity = arc_capacity - sum(demand_list[commodity_index] for commodity_index in fixed_pattern)

    if not "dimension_ratio" in dico_info: dico_info["dimension_ratio"] = []
    dico_info["dimension_ratio"].append(len(variable_pattern) / nb_commodities)

    # call to the separation/decomposition algorithm
    if iterative_separation:
        constraint_coeff, pre_pattern_and_amount_list = in_out_separation_decomposition_iterative(variable_demand_list, variable_outter_flow_per_commodity, variable_inner_flow_per_commodity, remaining_arc_capacity)
    else:
        constraint_coeff, pre_pattern_and_amount_list = in_out_separation_decomposition(variable_demand_list, variable_outter_flow_per_commodity, variable_inner_flow_per_commodity, remaining_arc_capacity)

    variable_commodity_coeff_list, constant_coeff = constraint_coeff
    pattern_and_amount_list = [([variable_pattern[index] for index in pattern] + fixed_pattern, amount) for pattern, amount in pre_pattern_and_amount_list]
   
    # lifting of the cut's coefficients
    commodity_coeff_list, constant_coeff, lifting_pattern_list = compute_all_lifted_coefficients(demand_list, variable_pattern, variable_commodity_coeff_list, fixed_pattern, constant_coeff, remaining_arc_capacity)

    for pattern in lifting_pattern_list:
        pattern_and_amount_list.append((pattern, 0))

    return (commodity_coeff_list, constant_coeff), pattern_and_amount_list

def knapsack_solver(weight_list, capacity, value_list, precision=10**-7):
    # this function solves a classical knapsack problem by calling a MINKNAP algorithm coded in c++ in an external library
    nb_objects = len(value_list)

    if capacity <= 0:
        return [0] * nb_objects, -10**10

    value_list_rounded = (np.array(value_list)/ precision).astype(int)

    instance = knapsacksolver.Instance()
    instance.set_capacity(capacity)

    for object_index in range(nb_objects):
        instance.add_item(weight_list[object_index], value_list_rounded[object_index])

    solution = knapsacksolver.solve(instance, algorithm = "minknap", verbose = False)

    return [object_index for object_index in range(nb_objects) if solution.contains(object_index)], solution.profit() * precision


def approximate_knapsack_optimizer(demand_list, arc_capacity, objective_coeff_per_commodity):
    # heuristic greedy resolution of the penalized knapsack problem considered in the paper
    nb_commodities = len(demand_list)
    order_list = [(objective_coeff_per_commodity[commodity_index] / demand_list[commodity_index], commodity_index) for commodity_index in range(nb_commodities)]
    order_list.sort()
    remaining_arc_capacity = arc_capacity
    value = 0
    pattern = []

    while order_list != []:
        ratio, commodity_index = order_list.pop()

        if ratio <= 0:
            break

        elif demand_list[commodity_index] <= remaining_arc_capacity:
            value += objective_coeff_per_commodity[commodity_index]
            remaining_arc_capacity -= demand_list[commodity_index]
            pattern.append(commodity_index)

    return pattern, value




nb_calls = 0
def penalized_knapsack_optimizer(demand_list, arc_capacity, objective_coeff_per_commodity, overload_penalization=1, verbose=0):
    # this function solves a special knapsack problem where over-capacitating the knapsack is allowed but penalised
    # this problem can be solved by solving two classical knapsack problem (see the paper for details)
    nb_commodities = len(demand_list)
    total_demand = sum(demand_list)
    global nb_calls
    nb_calls += 1

    first_solution, first_solution_value = knapsack_solver(demand_list, arc_capacity, np.array(objective_coeff_per_commodity))

    value_array = overload_penalization * np.array(demand_list) - np.array(objective_coeff_per_commodity)
    value_list = np.array(value_array)
    weight_list = np.array(demand_list)

    mask = value_list >= 0
    value_list *= mask
    weight_list = weight_list * mask + (1 - mask) * 2*(total_demand - arc_capacity)

    second_solution, second_solution_value = knapsack_solver(weight_list, total_demand - arc_capacity, value_list)
    second_solution_value = second_solution_value + overload_penalization * arc_capacity - sum(value_array)

    if first_solution_value >= second_solution_value:
        solution = first_solution 
        solution_value = first_solution_value

    else:
        is_in_solution = np.ones(nb_commodities)
        is_in_solution[second_solution] = 0
        solution = [commodity_index for commodity_index in range(nb_commodities) if is_in_solution[commodity_index]]
        solution_value = second_solution_value
        
    solution_overload = overload_penalization *  max(0, sum(np.array(demand_list)[solution]) - arc_capacity)

    return solution, solution_value, solution_overload


def approximate_penalized_knapsack_optimizer(demand_list, arc_capacity, objective_coeff_per_commodity, overload_penalization=1):
    # heuristic greedy resolution of the penalized knapsack problem considered in the paper
    nb_commodities = len(demand_list)
    order_list = [(objective_coeff_per_commodity[commodity_index] / demand_list[commodity_index], commodity_index) for commodity_index in range(nb_commodities)]
    order_list.sort()
    remaining_arc_capacity = max(0, arc_capacity)
    value = overload_penalization * min(0, arc_capacity)
    pattern = []

    while order_list != []:
        ratio, commodity_index = order_list[-1]

        if ratio <= 0:
            break

        elif demand_list[commodity_index] <= remaining_arc_capacity:
            value += objective_coeff_per_commodity[commodity_index]
            remaining_arc_capacity -= demand_list[commodity_index]
            pattern.append(commodity_index)
            order_list.pop()

        else:
            break

    gained_value_list = order_list

    while gained_value_list != []:

        l = []
        for _, commodity_index in gained_value_list:
            objective_coeff, demand = objective_coeff_per_commodity[commodity_index], demand_list[commodity_index]
            l.append((objective_coeff - overload_penalization * max(0, demand - remaining_arc_capacity), commodity_index))
        gained_value_list = l

        gained_value, commodity_index = max(gained_value_list)

        if gained_value <= 0:
            break

        else:
            value += gained_value
            remaining_arc_capacity = max(0, remaining_arc_capacity - demand_list[commodity_index])
            pattern.append(commodity_index)
            gained_value_list.remove((gained_value, commodity_index))

    pattern_overload = overload_penalization *  max(0, sum(np.array(demand_list)[pattern]) - arc_capacity)
    return pattern, value, pattern_overload
