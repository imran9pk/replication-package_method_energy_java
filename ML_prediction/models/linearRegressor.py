from sklearn.linear_model import LinearRegression

def lr_regressor(X_train, X_test, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model
