from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

def ada(hyperparameters, X_train, X_test, y_train, randomState, task_type):

    if task_type == 'classification':
        model = AdaBoostClassifier(random_state=randomState)
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = model.predict_proba(X_test)
    else:
        # Using thresholdVariance Feature Selection
        # {'n_estimators': 100, 'loss': 'linear', 'learning_rate': 0.1}
        if hyperparameters:
            n_estimators = 100
            loss = 'linear'
            learning_rate = 0.1
            model = AdaBoostRegressor(n_estimators=n_estimators, loss=loss, learning_rate=learning_rate, random_state=randomState)
        else:
            model = AdaBoostRegressor(random_state=randomState)

        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = None

    return y_predictions, y_predictions_proba, model
