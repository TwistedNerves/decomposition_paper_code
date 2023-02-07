# decomposition_paper_code

This repository is the code base of the paper "On the integration of Dantzig-Wolfe and Fenchel decompositions via directional normalizations"
It is not made to be understandable without reading the paper first.

To run the code you will need Python 3 with common libraries (matplotlib, numpy...), the mathematical programming solver Gurobi, and an external library solving the knapsack problem in C++ available at https://github.com/fontanf/knapsacksolver (requires Bazel for its compilation, see fontanf's repository)



Content of the files

knapsacksolver : copy of fontanf's repository in case it is not available anymore

instance_files_decomposition.py : folder containing all the datasets of unsplittable flow instances used in the experiments of the paper, also contains the result_files obtained by applying the decomposition methods on the datasets.

instance_mcnf.py : contains all the function to create instances of the unsplittable flow problem

create_and_store_instances_decomposition.py : code used to automatically generate all the datasets above, can be parametrized inside the file to create new instances

decompositon_methods.py : file containing the decomposition methods presented in the paper. Especially, Dantzig-Wolfe decomposition is implemented in knapsack_model_solver. Fenchel decomposition and the new decomposition presented in the paper are implemented in run_DW_Fenchel_model. The other function are building blocks of those two.

knapsack_oracles.py : contains algorithms that implement both optimization oracles and separation oracles for the special knapsack polyhedron appearing as the capacity constraints of the unsplittable flow problem (see the paper). This oracles are used as the subproblems of the decomposition methods implemented in decompositon_methods.py

mcnf_do_test.py : can be used to quicly test the decomposition methods, can create instances (parameters are to be chosen in the file) and a list of algorithm to apply can be selected

k_shortest_path.py : implementation of a k-shortest path algorithm. Used to compute allowed paths for each commodity of the unsplittable flow problem (so that no column generation is required to solve the path formulation (see paper for details)

launch_dataset_test_decomposition : code used to launch the resolution of the unsplittable flow datasets by the decomposition methods, uses multiprocessing, stores the results in the folder of the dataset

plot_results_decompostion.py : code used to analyse the results created by launching launch_dataset_test_decomposition and plots the results as shown in the paper

knapsacksolver.so : result of the compilation of the C++ library for the knapsack solver. This library needs to be compiled with Bazel (see fontanf's repository) this result files should be copied at the root file of the decomposition_paper_code folder.
