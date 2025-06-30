from scipy.stats import randint, uniform

regression_grids = {
    "RF": {
        "n_estimators": randint(100, 1000),
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2"]
    },
    "GB": {
        "n_estimators": randint(100, 1000),
        "max_depth": [3, 5, 10],
        "learning_rate": uniform(0.01, 0.3)
    },
    "DT": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5)
    },
    "KNN": {
        "n_neighbors": randint(3, 15),
        "weights": ["uniform", "distance"],
        "leaf_size": randint(20, 40)
    }
}
