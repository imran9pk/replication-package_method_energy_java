from business import main
from utils import data_preparation, results, featureSelection
from utils.TeeLogger import TeeLogger
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import RandomizedSearchCV
from pathlib import Path
import sys
from scipy.stats import randint
import numpy as np
from utils.models import ModelType
from utils import models
from utils import hyperParameterGrids
from utils import shapCalculator


## Paths definition
path = Path.cwd()
results_dir = path / "results"
current_config_name = "resultsHyperParameter_varianceThreshold"

result_file_simple = results_dir / "resultsHyperParamter.csv"
log_path = results_dir / f"{current_config_name}.log"

# Start logging
sys.stdout = TeeLogger(log_path)

#features
columns = []

############################# PARAMS #############################

#Train/Test size
test_size = 0.2 #20% test 80% train

#Random State
random_state = 42

#Models
available_models = [ModelType.RandomForestRegressor, ModelType.ADABoostRegressor]

##### Sampling #####
#sampling_mode = ["None","Over","Under"]
sampling_mode = ["None"]

##### HyperParameter #####
hyperparameters_dictionary = {}
hyperparameters = True

##### AutoSpearman #####
autoSpearman = False

##### RFECV Feature Selection #####
featureSelectionVar = False

##### KBest Feature Selection #####
featureSelectionKbest = False
top_k=30

##### Low Variance Drop #####
lowVarianceDrop = True

##### Drop Execution Time #####
dropExecTime = False

##### SHAP Feature Analysis #####
SHAP = True

##### Task Type #####
task_type = "regression" #classification/regression

############################# END PARAMS ######################


############################# RUN #############################

##### Data Preparation #####
#LOADING Data From CSV
X_Data, Y_Data, df, loaded_dataset_name = main.dataLoadingPandas(columns, lowVarianceDrop, dropExecTime, task_type)

##### Auto SpearMan Feature Selection #####
if(autoSpearman):
    X_Data = featureSelection.autoSpearman(X_Data)
    print("Columns selected by Auto Spearman:")
    columns = X_Data.columns
    print(columns)
    #### Reloading Data with a different set of features####
    X_Data, Y_Data, df, loaded_dataset_name = main.dataLoadingPandas(columns, lowVarianceDrop, dropExecTime, task_type)

##### Feature Selection RFECV #####
if(featureSelectionVar):
    columns = featureSelection.rfecv(X_Data, Y_Data, task_type)
    print("Columns selected by RFECV:")
    print(columns)
    #### Reloading Data with a different set of features####
    X_Data, Y_Data, df, loaded_dataset_name = main.dataLoadingPandas(columns, lowVarianceDrop, dropExecTime, task_type)

##### Feature Selection KBest #####
if(featureSelectionKbest):
    X_Data, columns = featureSelection.select_features_kbest(X_Data, Y_Data, task_type=task_type, k=top_k)
    print(f"Columns selected by KBest{top_k}:")
    print(columns)
    #### Reloading Data with a different set of features####
    X_Data, Y_Data, df, loaded_dataset_name = main.dataLoadingPandas(columns, lowVarianceDrop, dropExecTime, task_type)


#################### Prediction and Results #################################
configuration = [loaded_dataset_name, lowVarianceDrop, dropExecTime, featureSelectionVar, featureSelectionKbest, autoSpearman, hyperparameters, str(len(X_Data.columns))]

###### Ignition ######
for sMode in sampling_mode:

    x_Train, x_Test, y_Train, y_Test = data_preparation.prepare(X_Data, Y_Data, test_size, sMode, task_type)
    print(f"y_Test shape: {y_Test.shape}")
    print(y_Test.describe())
    print(f"y_Test dtype: {y_Test.dtype}")

##############################HYper Parameter Tuning##############################

    if hyperparameters:
        print("Hyperparameter tuning is enabled.")
        
        for a_model_type in available_models:
            try:
                # Hyperparameter Tuning - Setup RandomizedSearchCV to find the best hyperparameters for the model.
                print(f"Running RandomizedSearchCV for {a_model_type.value}...")
                model = models.get_model(a_model_type)
                param_grid = hyperParameterGrids.get_hyperparameter_grid(a_model_type)
                scoring_metric = hyperParameterGrids.get_scoring_metric(a_model_type)

                rf_random = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring=scoring_metric, n_iter=2500, cv=5, verbose=2, random_state=42, n_jobs=-1)
                
                search = rf_random.fit(x_Train, y_Train)

                # Retrieve the best model from the search.
                best_model = search.best_estimator_

                # Get the best hyperparameters found by RandomizedSearchCV.
                best_params = search.best_params_
                hyperparameters_dictionary[a_model_type] = best_params

                # Log the best hyperparameters found by RandomizedSearchCV.            
                print(f"Best hyperparameters for {a_model_type.value}: {best_params}")

                # Test the best model on the test set.
                y_pred = best_model.predict(x_Test)

                if task_type == "regression":
                    y_predictions_proba = None

                if(SHAP):
                    shapCalculator.calculateShap(best_model, x_Test, task_type, configuration, False, None, model_name=a_model_type.value)

                results.display_hyper_results(y_Test , y_pred, y_predictions_proba, a_model_type.value, result_file_simple, configuration, sMode, task_type)
            except Exception as e:
                print(f"Error during hyperparameter tuning for {a_model_type.value}: {e}")
                continue

# Cleanly close and restore stdout
sys.stdout.log.close()     
sys.stdout = sys.__stdout__ 