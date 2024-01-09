from src.launch_dataset import launch_dataset

# Set the path to the repository folder
global_path = "/home/sureli/Documents/Francois"
# assert False, "Unassigned global_path : Complete global_path with the path to the main directory"

nb_workers = 10 # for multiprocessing
duration_before_timeout = 60*60 # time limit given to the algorithms

# List the algorithm lauched on the dataset
# dataset_name, algorithm_list = "graph_scaling_dataset_lower_bound", ["DW-Fenchel iterative", "DW interior"]
# dataset_name, algorithm_list = "small_dataset", ["Fenchel", "Fenchel no preprocessing", "DW-Fenchel", "DW-Fenchel iterative", 'DW-Fenchel no preprocessing', 'DW', "DW momentum", "DW interior"]
# dataset_name, algorithm_list = "small_dataset_lower_bound", ["Fenchel", "DW-Fenchel", "DW-Fenchel iterative", 'DW', "DW momentum", "DW interior"]
# dataset_name, algorithm_list = "graph_scaling_dataset", ["Fenchel", "DW-Fenchel", "DW-Fenchel iterative", 'DW', "DW momentum", "DW interior"]
# dataset_name, algorithm_list = "capacity_scaling_dataset", ["DW-Fenchel iterative", "DW interior"]

dataset_name, algorithm_list = "smallest_dataset", ['DW', "DW momentum", "DW in out", "DW interior", "Fenchel", "DW-Fenchel", "DW-Fenchel iterative"]
launch_dataset(global_path, dataset_name, algorithm_list, nb_workers, duration_before_timeout)

dataset_name, algorithm_list = "smallest_dataset_path_gen", ['DW', "DW momentum", "DW in out", "DW interior", "Fenchel", "DW-Fenchel", "DW-Fenchel iterative"]
launch_dataset(global_path, dataset_name, algorithm_list, nb_workers, duration_before_timeout, path_generation_loop=True)
