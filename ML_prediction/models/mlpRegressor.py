from sklearn.neural_network import MLPRegressor

def mlp_regressor(X_train, X_test, y_train, random_state=42):
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model
