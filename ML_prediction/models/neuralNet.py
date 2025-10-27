from sklearn.neural_network import MLPClassifier, MLPRegressor

def net(X_train, X_test, y_train, randomState, task_type):

    if task_type == 'classification':
        model = MLPClassifier(max_iter=100000, random_state=randomState)
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = model.predict_proba(X_test)
    else:  # regression
        model = MLPRegressor(max_iter=100000, random_state=randomState)
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = None

    return y_predictions, y_predictions_proba, model
