from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def DT(hyperparameters, X_train, X_test, y_train, criterion, max_depth, max_features, min_samples_leaf, random_state, task_type):

    if task_type == 'classification':
        if not hyperparameters:
            model = DecisionTreeClassifier(random_state=random_state)
        else:
            model = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = model.predict_proba(X_test)

    else:  # regression
        if not hyperparameters:
            model = DecisionTreeRegressor(random_state=random_state)
        else:
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = None  # not available for regression

    return y_predictions, y_predictions_proba, model
