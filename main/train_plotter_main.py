

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
    pruned_name_7: str = "pruned_5"
    

    plot_accuracy_curves(
        csv_paths=[unpruned, pruned, pruned_5, pruned_7, base_line_best],
        names=[unpruned_name, pruned_name, pruned_name_5, pruned_name_7, base_line_best_pruned_name], 
        plot_name="With_07.png"
    )