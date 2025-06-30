from sklearn.neighbors import KNeighborsRegressor

def knn_regressor(X_train, X_test, y_train):
    model = KNeighborsRegressor(n_neighbors=3, weights='distance')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model
