from utility.model_train_plotter import plot_accuracy_curves


if __name__ == "__main__":

    unpruned: str = "EXPERIMENTS\\CONTINUATION_BEST_03_RIGHT\\Graph_ENSEMBLE_logs\\Graph\\version_2\\metrics.csv"
    unpruned_name: str = "unpruned"

    pruned: str = "EXPERIMENTS\\PRUNED_BEST_03_RIGHT\\Graph_ENSEMBLE_logs\\Graph\\version_0\\metrics.csv"
    pruned_name: str = "pruned"

    pruned_5: str = "EXPERIMENTS\\PRUNED_BEST_05_RIGHT\\Graph_ENSEMBLE_logs\\Graph\\version_0\\metrics.csv"
    pruned_name_5: str = "pruned_5"

    weighted: str = "EXPERIMENTS\\WEIGHTED_BEST_03_RIGHT\\Graph_ENSEMBLE_logs\\Graph\\version_5\\metrics.csv"
    weighted_name: str = "weighted"

    softmax_weighted: str = "EXPERIMENTS\\WEIGHTED_SOFTMAX_BEST_03_RIGHT\\Graph_ENSEMBLE_logs\\Graph\\version_0\\metrics.csv"
    softmax_weighted_name: str = "softmax weighted"


    softmax_weighted_pruned: str = "EXPERIMENTS\\WEIGHTED_PRUNED_BEST_03_RIGHT\\Graph_ENSEMBLE_logs\\Graph\\version_0\\metrics.csv"
    softmax_weighted_pruned_name: str = "softmax weighted pruned"

    base_line_best: str = "EXPERIMENTS\\BASELINE_BEST_03_RIGHT\\Graph_ENSEMBLE_logs\\Graph\\version_4\\metrics.csv"
    base_line_best_pruned_name: str = "baseline best"

    pruned_7: str = "EXPERIMENTS\\PRUNED_BEST_07_RIGHT\\Graph_ENSEMBLE_logs\\Graph\\version_0\\metrics.csv"
    pruned_name_7: str = "pruned_7"

    pruned_9: str = "EXPERIMENTS\\PRUNED_BEST_09_RIGHT\\Graph_ENSEMBLE_logs\\Graph\\version_0\\metrics.csv"
    pruned_name_9: str = "pruned_9"

    unpruned_3 = "EXPERIMENTS\\model_name\\Graph_ENSEMBLE_logs\\Graph\\version_0\\metrics.csv"
    unpruned_3_name = "unpruned_3"

    unpruned_5 = "EXPERIMENTS\\model_name\\Graph_ENSEMBLE_logs\\Graph\\version_1\\metrics.csv"
    unpruned_5_name = "unpruned_5"

    unpruned_7 = "EXPERIMENTS\\model_name\\Graph_ENSEMBLE_logs\\Graph\\version_2\\metrics.csv"
    unpruned_7_name = "unpruned_7"

    unpruned_9 = "EXPERIMENTS\\model_name\\Graph_ENSEMBLE_logs\\Graph\\version_3\\metrics.csv"
    unpruned_9_name = "unpruned_9"

    pruned_7_max_acc: str = "EXPERIMENTS\\UNPRUNED_BEST_07_RIGHT_max_acc\\Graph_ENSEMBLE_logs\\Graph\\version_0\\metrics.csv"
    pruned_name_7_max_acc: str = "pruned_7_.15_max_acc"

    pruned_9_max_acc: str = "EXPERIMENTS\\UNPRUNED_BEST_07_RIGHT_max_acc\\Graph_ENSEMBLE_logs\\Graph\\version_1\\metrics.csv"
    pruned_name_9_max_acc: str = "pruned_7_.2_max_acc"

    #plot_accuracy_curves(
    #    csv_paths=[unpruned, pruned, pruned_5, pruned_7, pruned_9, base_line_best],
    #    names=[unpruned_name, pruned_name, pruned_name_5, pruned_name_7, pruned_name_9, base_line_best_pruned_name], 
    #    plot_name="With_09.png"
    #)

    plot_accuracy_curves(
        csv_paths=[pruned_7_max_acc, pruned_9_max_acc, base_line_best],
        names=[pruned_name_7_max_acc, pruned_name_9_max_acc, base_line_best_pruned_name], 
        plot_name="With_08_pruned_max_acc.png"
    )