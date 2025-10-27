from sklearn.linear_model import LogisticRegression, LinearRegression

def LR(X_train, X_test, y_train, randomState, task_type):

    if task_type == 'classification':
        model = LogisticRegression(max_iter=100000, random_state=randomState)
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = model.predict_proba(X_test)

    else:  # regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = None  # no probability for regression

    return y_predictions, y_predictions_proba, model
