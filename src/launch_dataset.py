# import random
import numpy as np
import time
import pickle
from multiprocessing import Process, Manager

from src.decomposition_methods import *

def launch_dataset(global_path, dataset_name, algorithm_list, nb_workers, duration_before_timeout, path_generation_loop=False, nb_repetitions=1):
    # Launches all the algorithms to test on the instance present in the dataset directory
    # The number of time algorithms are lauched is decided with nb_repetitions

    # Open the file containing the name of the instances
    instance_name_file = open(global_path + "/decomposition_paper_code/instance_files_decomposition/" + dataset_name + "/instance_name_file.p", "rb" )
    instance_name_list = pickle.load(instance_name_file)
    instance_name_file.close()

    log_file = open(global_path + "/decomposition_paper_code/log_file.txt", 'w')
    log_file.write("Start\n")
    log_file.close()

    manager = Manager()
    result_dict = {algorithm_name : {instance_name : [None]*nb_repetitions for instance_name in instance_name_list} for algorithm_name in algorithm_list}
    worker_list = []
    # list containing the task to be done
    computation_list = [(repetition_index, instance_index, instance_name, algorithm_name) for repetition_index in range(nb_repetitions)
                                                                                            for instance_index, instance_name in enumerate(instance_name_list)
                                                                                                for algorithm_name in algorithm_list]

    # main loop : checking if processes are finished, creating new ones ...
    while len(computation_list) + len(worker_list) > 0:

        remaining_worker_list = []
        for process, start_time, return_list, computation_info in worker_list:
            repetition_index, instance_index, instance_name, algorithm_name = computation_info

            if not process.is_alive(): # if a process has ended, store the results
                result_dict[algorithm_name][instance_name][repetition_index] = (list(return_list), time.time() - start_time)

            elif time.time() > start_time + duration_before_timeout: # if a process has been running for more than the allowed time, kills it and store the results
                process.terminate()
                result_dict[algorithm_name][instance_name][repetition_index] = (list(return_list), duration_before_timeout)

            else: # else, keep the process in the worker list
                remaining_worker_list.append((process, start_time, return_list, computation_info))

        worker_list = remaining_worker_list

        # create a process if there are still task to be done and if there is space in the worker list
        if len(worker_list) < nb_workers and len(computation_list) > 0:
            computation_info = computation_list.pop(0)
            repetition_index, instance_index, instance_name, algorithm_name = computation_info

            print_string = "repetition : {0}/{1}, instance : {2}/{3}, algorithm : {4}".format(repetition_index, nb_repetitions, instance_index, len(instance_name_list), algorithm_name)
            instance_file_path = global_path + "/decomposition_paper_code/instance_files_decomposition/" + dataset_name + "/" + instance_name + ".p"
            return_list = manager.list()

            # lauching the process on a task
            process = Process(target=launch_solver_on_instance, args=(instance_file_path, algorithm_name, print_string, global_path, path_generation_loop, return_list))
            start_time = time.time()
            process.start()
            worker_list.append((process, start_time, return_list, computation_info))


    # Write the results in a file
    result_file = open(global_path + "/decomposition_paper_code/instance_files_decomposition/" + dataset_name + "/result_file.p", "wb" )
    pickle.dump(result_dict, result_file)
    result_file.close()

    import datetime
    log_file = open(global_path + "/decomposition_paper_code/log_file.txt", 'a')
    log_file.write(datetime.datetime.now().__str__())
    log_file.close()


def launch_solver_on_instance(instance_file_path, algorithm_name, print_string, global_path, path_generation_loop, return_list):
    # Lauch the algorithm named algortihm_name on the instance store in the file at instance_file_path

    print(print_string)

    # Read the instance in the instance file
    instance_file = open(instance_file_path, "rb" )
    graph, commodity_list, solution = pickle.load(instance_file)
    instance_file.close()

    total_demand = sum([commodity[2] for commodity in commodity_list])
    nb_commodities = len(commodity_list)
    nb_nodes = len(graph)

    #  creates a list of usable path for each commodity: the 4 shortest paths for each commodity
    possible_paths_per_commodity = compute_possible_paths_per_commodity(graph, commodity_list, 4)
    for commodity_index in range(nb_commodities):
        possible_paths_per_commodity[commodity_index].append(solution[commodity_index])


    temp = time.time()

    # Launch the chosen algorithm
    if algorithm_name == "DW" : knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, stabilisation="", path_generation_loop=path_generation_loop, verbose=0)
    if algorithm_name == "DW momentum" : knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, stabilisation="momentum", path_generation_loop=path_generation_loop, verbose=0)
    if algorithm_name == "DW in out" : knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, stabilisation="in_out", path_generation_loop=path_generation_loop, verbose=0)
    if algorithm_name == "DW interior" : knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, stabilisation="interior_point", path_generation_loop=path_generation_loop, verbose=0)
    if algorithm_name == "Fenchel" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(False, True, False), path_generation_loop=path_generation_loop, verbose=0)
    if algorithm_name == "Fenchel no preprocessing" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(False, False, False), path_generation_loop=path_generation_loop, verbose=0)
    if algorithm_name == "DW-Fenchel" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(True, True, False), path_generation_loop=path_generation_loop, verbose=0)
    if algorithm_name == "DW-Fenchel no preprocessing" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(True, False, False), path_generation_loop=path_generation_loop, verbose=0)
    if algorithm_name == "DW-Fenchel iterative" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(True, True, True), path_generation_loop=path_generation_loop, verbose=0)

    computing_time = time.time() - temp

    log_file = open(global_path + "/decomposition_paper_code/log_file.txt", 'a')
    log_file.write("Finished : " + instance_file_path + ", " + print_string + "\n")
    log_file.close()

    print("Finished")
