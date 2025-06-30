from sklearn.tree import DecisionTreeRegressor

def dt_regressor(hyperparameters, X_train, X_test, y_train, criterion=None, max_depth=None,
                 max_features=None, min_samples_leaf=1, random_state=42):

    if not hyperparameters:
        model = DecisionTreeRegressor(random_state=random_state)
    else:
        model = DecisionTreeRegressor(
            criterion=criterion if criterion else "squared_error",
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model
