from sklearn.ensemble import RandomForestRegressor

def rf_regressor(hyperparameters, X_train, X_test, y_train, max_depth=None, n_estimators=100, max_features='auto', min_samples_leaf=1, random_state=42):

    if not hyperparameters:
        model = RandomForestRegressor(random_state=random_state)
    else:
        model = RandomForestRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred, model
