from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

def gb_regressor(X_train, X_test, y_train, random_state=42):
    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model

def hgb_regressor(X_train, X_test, y_train, random_state=42):
    model = HistGradientBoostingRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model
