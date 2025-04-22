import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

dico_info = {}
dico_patterns = {}


def mean_confidence_interval_bootstrap(data, confidence=0.95, nb_iterations=100):
    # Computes the mean and confidence interval of the the input data array-like using the bootstraping method

    data = 1.0 * np.array(data)
    size = len(data)
    mean = np.mean(data)

    mean_list =[]
    for i in range(nb_iterations):
        sample = np.random.choice(data, size=size, replace=True)
        mean_list.append(np.mean(sample))

    mean_list.sort()
    upper_confidence_interval_bound = mean_list[int(nb_iterations * confidence + 0.5)]
    lower_confidence_interval_bound = mean_list[int(nb_iterations * (1 - confidence) + 0.5)]

    return mean, lower_confidence_interval_bound, upper_confidence_interval_bound


def median_and_quantile(data, quantile=0.8):
    # Computes the mean and confidence interval of the the input data array-like using the bootstraping method

    median = np.median(data)
    upper_quantile = np.quantile(data, quantile)
    lower_quantile = np.quantile(data, 1-quantile)

    return median, lower_quantile, upper_quantile


def ressample_curve(xy_list, new_abscisse_list):
    # a curve is given in xy_list with its abscisse and ordinate
    # return the same curve but define with different points, the new abscisse to use is given in new_abscisse_list
    new_y_value_list = []

    index_sup = None
    for abscisse in new_abscisse_list:
        for index, xy in enumerate(xy_list):
            x, y = xy
            if x >= abscisse:
                index_sup = index
                break
        if index == 0:
             x1, y1 = xy_list[0]
             x2, y2 = xy_list[1]
        elif index is None:
             x1, y1 = xy_list[-2]
             x2, y2 = xy_list[-1]
        else:
             x1, y1 = xy_list[index - 1]
             x2, y2 = xy_list[index]

        new_y_value = y1 + (abscisse - x1)/(x2 - x1) * (y2 - y1)
        new_y_value_list.append(new_y_value)

    return new_y_value_list


def plot_dataset_time_bounds(global_path, dataset_name, size_to_consider, abscisse, algorithm_list=None, x_label="Temps de calcul (s)", y_label="Valeur décalée des bornes", legend_position="upper left"):
    # This function reads the results of a dataset, aggregates the results of instances with the same parameters and plots the curves

    result_file = open(global_path + "/decomposition_paper_code/instance_files_decomposition/" + dataset_name + "/result_file.p", "rb" )
    result_dict = pickle.load(result_file)
    result_file.close()

    if algorithm_list is None:
        algorithm_list = list(result_dict.keys())

    # Color for each algorithm
    colors = {"Fenchel" : '#1f77b4', "Fenchel no preprocessing" : '#1f77b4', "DW-Fenchel" : '#ffbf0e', "DW in out" : '#ff00ff',
                "DW-Fenchel iterative" : '#d62728', "DW-Fenchel single point" : "#eeee00", 'DW-Fenchel no preprocessing' : '#d62728', "DW" : '#9467bd',
                "DW momentum" : '#2ca02c', "DW interior" : "#000000"}

     # Line style for each algorithm
    formating = {"Fenchel" : '', "Fenchel no preprocessing" : '-o', "DW-Fenchel" : '',
                "DW-Fenchel iterative" : '', "DW-Fenchel single point" : '', 'DW-Fenchel no preprocessing' : '-o', "DW" : '',
                "DW momentum" : '', "DW interior" : "", "DW in out" : ""}


     # Name in legend for each algorithm
    label = {"Fenchel" : 'Fenchel', "Fenchel no preprocessing" : 'Fenchel no preprocessing', "DW-Fenchel" : 'DW-Fenchel',
                "DW-Fenchel iterative" : 'DW-Fenchel iterative', "DW-Fenchel single point" : 'DW-Fenchel single point',
                'DW-Fenchel no preprocessing' : 'DW-Fenchel no preprocessing', "DW" : 'DW',
                "DW momentum" : 'DW momentum', "DW interior" : "DW interior point", "DW in out" : "DW in out"}


    figure = plt.figure()
    plt.rcParams.update({'font.size': 13})
    ax = figure.gca()
    # plt.xscale("log")
    plt.yscale("symlog", linthresh=10**-0)
    # plt.xscale("symlog", linthresh=10**-0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # computing the probable true value of the Dantzig-Wolfe linear relaxation
    DW_bounds = {}
    for algorithm_name in algorithm_list:
        for instance_name in result_dict[algorithm_name]:
            size = int(instance_name.split('_')[2]) # use for the other datasets
            # size = int(instance_name.split('_')[3]) # use for capacity_scaling_dataset
            if size == size_to_consider:
                results_list = result_dict[algorithm_name][instance_name][0][0]
                last_ub, last_lb, finish_time = results_list[-1]
                if instance_name not in DW_bounds or DW_bounds[instance_name][0] > abs(last_ub - last_lb):
                    DW_bounds[instance_name] = (abs(last_ub - last_lb), (last_ub + last_lb)/2)

    # aggregating the results and ploting the curves
    for algorithm_name in algorithm_list:
        print(algorithm_name)
        results_temp = {}
        for instance_name in result_dict[algorithm_name]:
            size = int(instance_name.split('_')[2]) # use for the other datasets
            # size = int(instance_name.split('_')[3]) # use for capacity_scaling_dataset
            if size == size_to_consider:
                results_temp[instance_name] = result_dict[algorithm_name][instance_name][0][0]

        # results aggregation
        ub_list_list = []
        lb_list_list = []
        for instance_name in results_temp:
            results_list = results_temp[instance_name]
            ub_time_list = [(time, ub - DW_bounds[instance_name][1]) for ub, lb, time in results_list]
            ub_time_list.append((10**5, ub_time_list[-1][1]))
            lb_time_list = [(time, lb - DW_bounds[instance_name][1]) for ub, lb, time in results_list]
            lb_time_list.append((10**5, lb_time_list[-1][1]))
            ub_list_list.append(ressample_curve(ub_time_list, abscisse))
            lb_list_list.append(ressample_curve(lb_time_list, abscisse))

        for i, lb_list in enumerate(lb_list_list):
            new_lb_list = []
            best_lb = None
            for lb in lb_list:
                if best_lb is None or lb > best_lb:
                    best_lb = lb
                new_lb_list.append(best_lb)
            lb_list_list[i] = new_lb_list

        # plotting the curves
        ub_list = [median_and_quantile(x) for x in zip(*ub_list_list)]
        lb_list = [median_and_quantile(x) for x in zip(*lb_list_list)]
        # ub_list = [mean_confidence_interval_bootstrap(x) for x in zip(*ub_list_list)]
        # lb_list = [mean_confidence_interval_bootstrap(x) for x in zip(*lb_list_list)]
        plt.plot(abscisse, [x[0] for x in ub_list], formating[algorithm_name], label=label[algorithm_name], color=colors[algorithm_name], markevery=20)
        plt.fill_between(abscisse, [x[1] for x in ub_list], [x[2] for x in ub_list], alpha=0.25, facecolor=colors[algorithm_name], edgecolor=colors[algorithm_name])
        plt.plot(abscisse, [x[0] for x in lb_list], formating[algorithm_name], color=colors[algorithm_name], markevery=20)
        plt.fill_between(abscisse, [x[1] for x in lb_list], [x[2] for x in lb_list], alpha=0.25, facecolor=colors[algorithm_name], edgecolor=colors[algorithm_name])
        if legend_position is not None:
            ax.legend(loc=legend_position, framealpha=0.0)


    plt.show()


if __name__ == "__main__":
    global_path = "/home/francois/Desktop"
    # assert False, "Unassigned global_path : Complete global_path with the path to the main directory"

    # choice of the dataset to plot
    # dataset_name = "low_demand_max_dataset"
    dataset_name = "small_low_demand_max_dataset"
    # dataset_name = "high_demand_max_dataset"
    # dataset_name = "small_high_demand_max_dataset"
    # dataset_name = "size_of_capacity_dataset"
    # dataset_name = "smallest_low_demand_max_dataset"
    # dataset_name = "smallest_low_demand_max_dataset_path_gen"


    # abscisse used to resample the curves
    abscisse = list(range(1, 60*60, 10))

    # list of the algorithms to plot
    algorithm_list = []
    algorithm_list.append("Fenchel")
    # algorithm_list.append("Fenchel no preprocessing")
    algorithm_list.append("DW")
    algorithm_list.append("DW momentum")
    algorithm_list.append("DW in out")
    algorithm_list.append("DW interior")
    algorithm_list.append("DW-Fenchel")
    algorithm_list.append("DW-Fenchel iterative")
    # algorithm_list.append("DW-Fenchel no preprocessing")

    plot_dataset_time_bounds(global_path, dataset_name, 40, abscisse, algorithm_list=algorithm_list, legend_position='lower right', x_label="Computing time (s)", y_label="Deviation from optimal bound")

