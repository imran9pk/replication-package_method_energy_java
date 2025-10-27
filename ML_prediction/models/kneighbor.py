from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

def kNN(hyperparameters, X_train, X_test, y_train, algorithm, leaf_size, metric, n_neighbors, weights, task_type):

    if task_type == 'classification':
        if not hyperparameters:
            model = KNeighborsClassifier()
        else:
            model = KNeighborsClassifier(
                algorithm=algorithm,
                leaf_size=leaf_size,
                metric=metric,
                n_neighbors=n_neighbors,
                weights=weights
            )
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = model.predict_proba(X_test)

    else:  # regression
        if not hyperparameters:
            model = KNeighborsRegressor()
        else:
            model = KNeighborsRegressor(
                algorithm=algorithm,
                leaf_size=leaf_size,
                metric=metric,
                n_neighbors=n_neighbors,
                weights=weights
            )
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = None  # not applicable

    return y_predictions, y_predictions_proba, model
