from src.launch_dataset import launch_dataset

# Set the path to the repository folder
global_path = "/home/sureli/Documents/Francois"
# global_path = "/home/francois/Desktop/"
# assert False, "Unassigned global_path : Complete global_path with the path to the main directory"

nb_workers = 10 # for multiprocessing
duration_before_timeout = 60*60 # time limit given to the algorithms

# List the algorithm lauched on the dataset
# dataset_name, algorithm_list = "small_low_demand_max_dataset", ["Fenchel", "Fenchel no preprocessing", "DW-Fenchel", "DW-Fenchel iterative", 'DW-Fenchel no preprocessing', 'DW', "DW momentum", "DW interior"]
# dataset_name, algorithm_list = "small_high_demand_max_dataset", ["Fenchel", "DW-Fenchel", "DW-Fenchel iterative", 'DW', "DW momentum", "DW interior"]
#dataset_name, algorithm_list = "low_demand_max_dataset", ["Fenchel", "DW-Fenchel", "DW-Fenchel iterative", "DW interior"]
# dataset_name, algorithm_list = "size_of_capacity_dataset", ["DW-Fenchel iterative", "DW interior"]

# dataset_name, algorithm_list = "small_low_demand_max_dataset", ['DW', "DW momentum", "DW in out", "DW interior", "Fenchel", "DW-Fenchel", "DW-Fenchel iterative"]
#launch_dataset(global_path, dataset_name, algorithm_list, nb_workers, duration_before_timeout)

#dataset_name, algorithm_list = "small_low_demand_max_dataset", ['DW', "DW momentum", "DW in out", "DW interior", "Fenchel", "DW-Fenchel", "DW-Fenchel iterative"]
#launch_dataset(global_path, dataset_name, algorithm_list, nb_workers, duration_before_timeout)

#dataset_name, algorithm_list = "high_demand_max_dataset", ["Fenchel", "DW-Fenchel", "DW-Fenchel iterative", "DW interior"]
#launch_dataset(global_path, dataset_name, algorithm_list, nb_workers, duration_before_timeout)

# dataset_name, algorithm_list = "smallest_low_demand_max_dataset", ["DW interior", "Fenchel", "DW-Fenchel", "DW-Fenchel iterative"]
# launch_dataset(global_path, dataset_name, algorithm_list, nb_workers, duration_before_timeout, path_generation_loop=True)

dataset_name, algorithm_list = "smallest_low_demand_max_dataset_no_flow_pen", ["DW interior", "Fenchel", "DW-Fenchel", "DW-Fenchel iterative"]
launch_dataset(global_path, dataset_name, algorithm_list, nb_workers, duration_before_timeout)
