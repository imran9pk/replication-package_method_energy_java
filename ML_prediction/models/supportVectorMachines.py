from sklearn import svm
from sklearn.ensemble import BaggingClassifier, BaggingRegressor


def SVM(X_train, X_test, y_train, kernel, task_type):

    if task_type == 'classification':
        base_model = svm.SVC(
            decision_function_shape='ovr',
            kernel=kernel,
            cache_size=20000,
            probability=True  # Required for predict_proba()
        )
        model = BaggingClassifier(estimator=base_model)
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = model.predict_proba(X_test)
    else:  # regression
        base_model = svm.SVR(kernel=kernel, cache_size=20000)
        model = BaggingRegressor(base_estimator=base_model)
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = None

    return y_predictions, y_predictions_proba, model
