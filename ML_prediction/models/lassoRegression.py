from sklearn.linear_model import Lasso

def LassoR(X_train, X_test, y_train, randomState, task_type):
    if task_type != 'regression':
        raise ValueError("Lasso regression is only applicable for regression tasks.")
    
    model = Lasso(alpha=1.0, random_state=randomState)
    model.fit(X_train, y_train)
    y_predictions = model.predict(X_test)
    y_predictions_proba = None  # no probability for regression

    return y_predictions, y_predictions_proba, model
