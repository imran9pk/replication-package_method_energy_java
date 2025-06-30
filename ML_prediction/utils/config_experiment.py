# Global experiment settings (override in each runner if needed)

dataset_path = "data/methods_dataset.csv"
results_base_path = "results/"
test_size = 0.2
random_state = 42
available_models = ["RF", "DT", "GB", "HGB", "MLP", "KNN", "LR"]

# Feature selection file paths
autospearman_feature_file = "results/selected_features_spearman.csv"
rfecv_feature_file = "results/selected_features_rfecv.csv"


