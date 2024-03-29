import random
import heapq as hp
import numpy as np
import time


def generate_instance(graph_type, graph_generator_inputs, demand_generator_inputs, nb_capacity_modifitcations=0):
    """this function generates an intances according to the asked caracteristics :
    - first a graph is generated : a grid graph or a random graph
    - then commodities are created so that there exist a solution to the unsplittable flow problem

    Inputs:
    graph_type : should be "grid", "random" or "random_connected"
    graph_generator_inputs : tuple of parameters depending on the graph type, for "random_connected" sould be (nb_nodes, edge_probability, nb_origins, arc_capacity)
    demand_generator_inputs : dictionary of parameters, typically {"max_demand" : upper_bound_on_commodity_demand}

    Outputs:
    graph : A list (one entry per node) of dictionaries (one entry per neighbor)
    commodity_list : A list (one entry per commodity) of tuples (origin, destination, demand)
    path_list : A list (one entry per commodity) of paths, they form an optimal solution of the unsplittable flow instance
    origin_list : A list of all nodes that are origin of at least one commodity
    """

    if graph_type == "grid":
        reverse_graph_generator = generate_grid_reverse_graph
    elif graph_type == "random":
        reverse_graph_generator = generate_random_reverse_graph
    elif graph_type == "random_connected":
        reverse_graph_generator = generate_random_connected_reverse_graph
    else:
        assert False, "No generator for this type of graph is implemented, check your spelling or contribute"

    # graph generation
    reverse_graph, is_origin_list = reverse_graph_generator(*graph_generator_inputs)
    origin_list = [node for node, is_origin in enumerate(is_origin_list) if is_origin]

    # commodities generation
    commodity_list, path_list = generate_demand(is_origin_list, reverse_graph, **demand_generator_inputs)

    # the created graph was reversed to facilitate demand generation so we reverse it
    graph = [{neighbor : reverse_graph[neighbor][node] for neighbor in range(len(reverse_graph)) if node in reverse_graph[neighbor]} for node in range(len(reverse_graph))]

    for index in range(nb_capacity_modifitcations):
        origin = np.random.choice(origin_list)
        neighbor1, neighbor2 = np.random.choice(list(graph[origin].keys()), 2)
        capacity_change = min(1 + int(graph[origin][neighbor1]/100), graph[origin][neighbor1], graph[origin][neighbor2])
        graph[origin][neighbor1] += capacity_change
        graph[origin][neighbor2] -= capacity_change

    return graph, commodity_list, path_list, origin_list

def generate_grid_reverse_graph(nb_origins, nb_row_grid, nb_column_grid, nb_origin_connections, grid_link_capacity=15000, other_link_capacity=10000, local_connection_of_origin = False):
    # generates a grid graph with additional nodes conected to the grid and uniform capacities
    # the graph is reversed

    reverse_graph = []

    for i in range(nb_row_grid * nb_column_grid + nb_origins):
        reverse_graph.append({})

    # generates the grid
    for i in range(nb_row_grid):
         for j in range(nb_column_grid):
             reverse_graph[i + nb_row_grid * j][(i+1)%nb_row_grid + nb_row_grid * j] = grid_link_capacity
             reverse_graph[i + nb_row_grid * j][(i-1)%nb_row_grid + nb_row_grid * j] = grid_link_capacity
             reverse_graph[i + nb_row_grid * j][i + nb_row_grid * ((j+1)%nb_column_grid)] = grid_link_capacity
             reverse_graph[i + nb_row_grid * j][i + nb_row_grid * ((j-1)%nb_column_grid)] = grid_link_capacity

    # adding the additional nodes (the origins)
    if local_connection_of_origin:
        for d in range(nb_origins):
            origin = d + nb_row_grid * nb_column_grid
            square_size = int(np.ceil(np.sqrt(nb_origin_connections)))
            i = random.randint(0, nb_row_grid-1)
            j = random.randint(0, nb_column_grid-1)

            count = 0
            for k in range(square_size):
                for l in range(square_size):
                    if count < nb_origin_connections:
                        reverse_graph[(i+k)%nb_row_grid + nb_row_grid * ((j+l)%nb_column_grid)][origin] = other_link_capacity
                        count += 1

    else:
        for d in range(nb_origins):
            origin = d + nb_row_grid * nb_column_grid
            for k in range(nb_origin_connections):
                i = random.randint(0, nb_row_grid-1)
                j = random.randint(0, nb_column_grid-1)
                reverse_graph[i + nb_row_grid * j][origin] = other_link_capacity

    is_origin_list = [0] * nb_row_grid * nb_column_grid + [1] * nb_origins

    return reverse_graph, is_origin_list

def generate_random_reverse_graph(nb_nodes, edge_proba, nb_origins, arc_capacity):
    # generates a random graph with uniform capacities
    # the graph is reversed

    reverse_graph = [{} for i in range(nb_nodes)]
    is_origin_list = [0]*nb_nodes
    for node in np.random.choice(nb_nodes, nb_origins, replace=False):
        is_origin_list[node] = 1

    for node in range(nb_nodes):
        for neighbor in range(nb_nodes):
            if node != neighbor and random.random() < edge_proba:
                reverse_graph[node][neighbor] = arc_capacity

    return reverse_graph, is_origin_list

def generate_random_connected_reverse_graph(nb_nodes, edge_proba, nb_origins, arc_capacity):
    # generates a random graph with uniform capacities
    # the graph is reversed
    # the returned graph is always strongly connected
    # to do so a random covering tree is generated and then random arcs are added to match the desired sparsity

    reverse_graph = [{} for i in range(nb_nodes)]
    is_origin_list = [0]*nb_nodes
    for node in np.random.choice(nb_nodes, nb_origins, replace=False):
        is_origin_list[node] = 1

    not_root_set = set(range(nb_nodes))

    while len(not_root_set) > 0:
        initial_node = random.choice(tuple(not_root_set))
        reachable = [False]*nb_nodes
        reachable[initial_node] = True
        pile = [initial_node]

        while pile:
            current_node = pile.pop()
            for neighbor in reverse_graph[current_node]:
                if not reachable[neighbor]:
                    reachable[neighbor] = True
                    pile.append(neighbor)

        unreachable_nodes = [node for node in range(nb_nodes) if not reachable[node]]
        if len(unreachable_nodes) == 0:
            not_root_set.remove(initial_node)
        else:
            chosen_node = random.choice(unreachable_nodes)
            reverse_graph[initial_node][chosen_node] = arc_capacity

    current_nb_edge = sum([len(d) for d in reverse_graph])
    edge_proba -= current_nb_edge / (nb_nodes**2 - nb_nodes)
    for node in range(nb_nodes):
        for neighbor in range(nb_nodes):
            if node != neighbor and random.random() < edge_proba:
                reverse_graph[node][neighbor] = arc_capacity

    return reverse_graph, is_origin_list

def generate_demand(is_origin_list, reverse_graph, random_filling_of_origins=True, random_paths=True, max_demand=1500, delete_resuidal_capacity=False,
                    smaller_commodities=False, verbose=0):
    # generates the commodities so that there exist a solution to the unsplittable flow problem
    # To create one commodity :
    # a random node is chosen, all the origins attainable from the node are computed
    # one is randomly chosen with a random path toward it (created with depth first seach), create a commodity demand that can fit on the path

    residual_graph = [{neighbor : reverse_graph[node][neighbor] for neighbor in reverse_graph[node]} for node in range(len(reverse_graph))]
    commodity_list = []
    path_list = []
    possible_destination_nodes = 1 - np.array(is_origin_list)

    i = 0
    while True:
        if i%100 == 0: print(i, end='\r')
        i+=1

        # choosing a random none origin node
        destination = np.random.choice(len(is_origin_list), p=possible_destination_nodes / sum(possible_destination_nodes))

        # getting all attainable origins
        origin_list = get_availables_origins(residual_graph, destination, is_origin_list, random_paths)

        # finishing the algorithm when no origin is attainable
        if origin_list == []:
            possible_destination_nodes[destination] = 0
            if sum(possible_destination_nodes) == 0:
                if verbose:
                    print()
                    print("residual value is ",sum([sum(dct.values()) for dct in residual_graph]))
                    print("full value is ",sum([sum(dct.values()) for dct in reverse_graph]))

                if delete_resuidal_capacity:
                    for node, neighbor_dict in enumerate(reverse_graph):
                        reverse_graph[node] = {neighbor : neighbor_dict[neighbor] - residual_graph[node][neighbor] for neighbor in neighbor_dict}

                return commodity_list, path_list
            else:
                continue

        # choosing an origin
        if random_filling_of_origins:
            origin, path = origin_list[random.randint(0, len(origin_list)-1)]
        else:
            origin, path = min(origin_list, key=lambda x:x[0])

        # allocating the commodity in the graph
        min_remaining_capacity = min([residual_graph[path[node_index]][path[node_index+1]] for node_index in range(len(path)-1)])

        if smaller_commodities:
            used_capacity = random.randint(1, min(min_remaining_capacity, max_demand))
        else:
            used_capacity = min(min_remaining_capacity, random.randint(1, max_demand))

        for node_index in range(len(path)-1):
            residual_graph[path[node_index]][path[node_index+1]] -= used_capacity

        commodity_list.append((origin, destination, used_capacity))
        path.reverse()
        path_list.append(path)



def get_availables_origins(residual_graph, initial_node, is_origin_list, random_paths):
    # look for all the origins attainable from the initial_node and a path to each of this origins
    pile = [(initial_node, [initial_node])]
    visited = [0]*len(residual_graph)
    visited[initial_node] = 1
    origin_list = []

    while pile != []:
        if random_paths:
            current_node, path = pile.pop(random.randint(0, len(pile)-1))
        else:
            current_node, path = pile.pop(0)

        for neighbor in residual_graph[current_node]:
            if residual_graph[current_node][neighbor] > 0 and not visited[neighbor]:
                visited[neighbor] = 1

                if is_origin_list[neighbor]:
                    origin_list.append((neighbor, path + [neighbor]))

                else:
                    pile.append((neighbor, path + [neighbor]))

    return origin_list
