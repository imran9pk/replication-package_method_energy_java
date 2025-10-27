from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)

def GB(X_train, X_test, y_train, randomState, task_type):

    if task_type == 'classification':
        model = GradientBoostingClassifier(random_state=randomState)
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = model.predict_proba(X_test)
    else:
        model = GradientBoostingRegressor(random_state=randomState)
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = None

    return y_predictions, y_predictions_proba, model


def HGB(X_train, X_test, y_train, randomState, task_type):

    if task_type == 'classification':
        model = HistGradientBoostingClassifier(random_state=randomState)
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = model.predict_proba(X_test)
    else:
        model = HistGradientBoostingRegressor(random_state=randomState)
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = None

    return y_predictions, y_predictions_proba, model
