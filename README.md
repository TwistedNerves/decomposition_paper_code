# Introduction

This repository is the code base of the paper "On the integration of Dantzig-Wolfe and Fenchel decompositions via directional normalizations"
It is not made to be understandable without reading the paper first.

To run the code you will need Python 3 with common libraries (matplotlib, numpy, pickle, ...), the mathematical programming solver Gurobi, and an external library solving the knapsack problem in C++ available at https://github.com/fontanf/knapsacksolver (requires Bazel for its compilation, see fontanf's repository; although a precompiled version is already present in the current repository, this version works on some Ubuntu distributions)


# Main functions and examples

To create instances of the unsplittable flow problem, use the function generate_instance in src/instance_mcnf.py

To apply the decomposition methods on an instance, use the functions knapsack_model_solver and run_DW_Fenchel_model in src/decompositon_methods.py

An example of how to create an instance and solve it is given in example_test_algorithm.py

An example of how to create a dataset of instance is given in example_create_and_store_instances.py

An example of how to launch the algorithms on a dataset is given in example_launch_dataset.py


# Content of the repository

**instance_files_decomposition** : folder containing all the datasets of unsplittable flow instances used in the experiments of the paper. Each instance is stored in one file using the python package Pickle. Each dataset also contains the result_files obtained by applying the decomposition methods on the datasets.

**knapsacksolver** : copy of fontanf's repository in case it is not available anymore

**src** : contains all the source code of the project

plot_results_decompostion.py : code used to analyse the results of the algorithms on the datasets and make the plots shown in the paper


## Content of **src**

instance_mcnf.py : contains all the function to create instances of the unsplittable flow problem, the main entry point to the instance creation is generate_instance

decompositon_methods.py : file containing the decomposition methods presented in the paper. Especially, Dantzig-Wolfe decomposition is implemented in knapsack_model_solver. Fenchel decomposition and the new decomposition presented in the paper are implemented in run_DW_Fenchel_model. The other function are building blocks of those two.

models.py : contains the functions used to create the master models of the decomposition methods. The Dantzig-Wolfe master problem is the knapsack model and the Fenchel master problem is the arc-path model.

knapsack_oracles.py : contains algorithms that implement both optimization oracles and separation oracles for the special knapsack polyhedron appearing as the capacity constraints of the unsplittable flow problem (see the paper). This oracles are used as the subproblems of the decomposition methods implemented in decompositon_methods.py

k_shortest_path.py : implementation of a k-shortest path algorithm. Used to compute allowed paths for each commodity of the unsplittable flow problem (so that no column generation is required to solve the path formulation (see paper for details)

knapsacksolver.so : result of the compilation of the C++ library for the knapsack solver. This library needs to be compiled with Bazel (see fontanf's repository) this result files should be copied in the **src** folder.

