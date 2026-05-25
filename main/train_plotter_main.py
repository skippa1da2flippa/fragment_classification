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

    bpt_3 = "EXPERIMENTS\\FINAL_BPT_KL_0.3\\Graph_ENSEMBLE_logs\\Graph\\version_1\\metrics.csv"
    bpt_3_name = "bpt 0.3"

    bpt_7 = "EXPERIMENTS\\FINAL_BPT_KL_0.7\\Graph_ENSEMBLE_logs\\Graph\\version_0\\metrics.csv"
    bpt_7_name = "bpt 0.7"

    bpt_9 = "EXPERIMENTS\\FINAL_BPT_KL_0.9\\Graph_ENSEMBLE_logs\\Graph\\version_0\\metrics.csv"
    bpt_9_name = "bpt 0.9"

    bpt_9_max = "EXPERIMENTS\\FINAL_BPT_KL_0.9\\Graph_ENSEMBLE_logs\\Graph_with_min_entropy\\version_0\\metrics.csv"
    bpt_9_name_max = "bpt 0.9 max"

    bpt_random_9_a = "EXPERIMENTS\\FINAL_BPT_KL_0.9\\Graph_ENSEMBLE_logs\\Graph_with_random\\version_0\\metrics.csv"
    bpt_random_9_a_name = "bpt 0.9 random a)"

    bpt_random_9_b = "EXPERIMENTS\\FINAL_BPT_KL_0.9\\Graph_ENSEMBLE_logs\\Graph_with_random\\version_1\\metrics.csv"
    bpt_random_9_b_name = "bpt 0.9 random b)"

    bpt_random_9_c = "EXPERIMENTS\\FINAL_BPT_KL_0.9\\Graph_ENSEMBLE_logs\\Graph_with_random\\version_2\\metrics.csv"
    bpt_random_9_c_name = "bpt 0.9 random c)"

    pth_9_cout = "EXPERIMENTS\\FINAL_BPT_KL_0.9\\Graph_ENSEMBLE_logs\\Graph\\version_3\\metrics.csv"
    pth_9_cout_name = "bpt 0.9 cout"

    pth_9_max_acc = "EXPERIMENTS\\FINAL_BPT_KL_0.9\\Graph_ENSEMBLE_logs\\Graph\\version_4\\metrics.csv"
    pth_9_max_acc_name = "Max acc bpt 9"

    adaptive_per = "EXPERIMENTS\\FINAL_BPT_KL_0.9\\Graph_ENSEMBLE_logs\\Graph_with_random\\version_7\\metrics.csv"
    adaptive_per_name = "adaptive"

    #plot_accuracy_curves(
    #    csv_paths=[unpruned, pruned, pruned_5, pruned_7, pruned_9, base_line_best],
    #    names=[unpruned_name, pruned_name, pruned_name_5, pruned_name_7, pruned_name_9, base_line_best_pruned_name], 
    #    plot_name="With_09.png"
    #)

    plot_accuracy_curves(
        csv_paths=[bpt_9, adaptive_per],
        names=[bpt_9_name, adaptive_per_name], 
        plot_name="plots\\KL_GNN_BPT_6.png"
    )