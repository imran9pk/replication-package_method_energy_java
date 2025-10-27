from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import numpy as np

def rf(hyperparameters, X_train, X_test, y_train, max_depth, n_estimators, max_features, min_sample_leaf, random_state, task_type):

    if task_type == 'classification':
        if(hyperparameters==False):
            model = RandomForestClassifier(random_state=random_state)
        else:
            model = RandomForestClassifier(max_depth=max_depth, random_state=random_state, n_estimators=n_estimators, max_features=max_features, min_samples_leaf=min_sample_leaf)

        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = model.predict_proba(X_test)

        return y_predictions, y_predictions_proba, model
    
    elif task_type == 'regression':
        if not hyperparameters:
            model = RandomForestRegressor(random_state=random_state)
        else:
            # Using thresholdVariance Feature Selection
            #{'bootstrap': True, 'max_depth': 5, 'max_features': None, 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 1577}

            #Using no feature selection_all features
            #{'bootstrap': True, 'max_depth': 5, 'max_features': None, 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 1577}

            #Using kbest_30 Feature Selection
            # {'bootstrap': True, 'max_depth': 5, 'max_features': None, 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 1577}

            #Using autospearman Feature Selection
            # {'bootstrap': True, 'max_depth': 5, 'max_features': None, 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 1577}
            model = RandomForestRegressor(
                bootstrap=True,
                max_depth=5,
                n_estimators=1577,
                max_features=None,
                min_samples_leaf=4,
                min_samples_split=5,
                random_state=random_state
            )

        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        y_predictions_proba = None  # not applicable for regression

    return y_predictions, y_predictions_proba, model

