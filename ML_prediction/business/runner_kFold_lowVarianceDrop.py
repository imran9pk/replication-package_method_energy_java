from business import main
from utils import data_preparation, results, featureSelection
from sklearn.model_selection import StratifiedKFold, KFold
from pathlib import Path
import sys
from utils.TeeLogger import TeeLogger

## Paths definition
path = Path.cwd()
results_dir = path / "results"
current_config_name = "results5Fold_lowVarianceDrop"

result_file_simple = results_dir / "resultsCV.csv"
result_file_kFold = results_dir / f"{current_config_name}.csv"
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
# models = ["RF", "ADA"]
models = ["ADA", "DT", "GB", "HGB", "kNN", "LassoR", "LR", "MLP", "RF", "RidgeR", "SVM"]

##### Sampling #####
#sampling_mode = ["None","Over","Under"]
sampling_mode = ["None"]

##### kFold #####
kfold = True
nFolds = 5

##### HyperParameter #####
#hyperparameters = []
hyperparameters = False

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

##### k fold #################################################################
splits_xy = []

if kfold:
    # Set target column based on task type
    if task_type == 'classification':
        target_column = 'isBenchmarked'
        splitter = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=42)
        X_Data = df.drop(columns=target_column)
        Y_Data = df[target_column]
        split_iterator = splitter.split(X_Data, Y_Data)
        
    elif task_type == 'regression':
        target_column = 'log_energy'
        splitter = KFold(n_splits=nFolds, shuffle=True, random_state=42)
        X_Data = df.drop(columns=target_column)
        Y_Data = df[target_column]
        split_iterator = splitter.split(X_Data)  # Don't pass Y_Data for regression

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    for i, (train_indices, test_indices) in enumerate(split_iterator):
        x_train = X_Data.iloc[train_indices]
        y_train = Y_Data.iloc[train_indices]
        x_test = X_Data.iloc[test_indices]
        y_test = Y_Data.iloc[test_indices]

        fold_id = i
        splits_xy.append((x_train, y_train, x_test, y_test, fold_id))

else:
    splits_xy.append((None, None, None, None, None))

#############################################################################


#################### Prediction and Results #################################
configuration = [loaded_dataset_name, lowVarianceDrop, dropExecTime, featureSelectionVar, featureSelectionKbest, autoSpearman, hyperparameters, str(len(X_Data.columns))]
result_file = result_file_kFold if kfold else result_file_simple

for x_Train, y_Train, x_Test, y_Test, fold_id in splits_xy:

    ###### Ignition ######
    for sMode in sampling_mode:

        #TRAIN-TEST-SPLIT
        if(kfold):
            x_Train, y_Train = data_preparation.prepareKfold(x_Train, y_Train, sMode, task_type)
        else:
            x_Train, x_Test, y_Train, y_Test = data_preparation.prepare(X_Data, Y_Data, test_size, sMode, task_type)
            print(f"y_Test shape: {y_Test.shape}")
            print(y_Test.describe())
            print(f"y_Test dtype: {y_Test.dtype}")

    ##### Prediction ######

        for model in models:
            try:
                print(f"\n[Fold {fold_id}] Model: {model}, Sampling: {sMode}")
                y_test_pred , y_predictions, y_predictions_proba, model_name = main.predict(model, hyperparameters, x_Train,
                                                                                    x_Test, y_Train, y_Test, random_state, SHAP, configuration, kfold, fold_id, task_type)
                print(f"Model {model} predictions completed.")
                ##### Display Data #####
                
                results.display(y_test_pred , y_predictions, y_predictions_proba, model_name, result_file, configuration, sMode, task_type, kfold, fold_id)
            except (Exception, ValueError) as e:
                print(f"Error with model {model}: {e}")
                continue

# Cleanly close and restore stdout
sys.stdout.log.close()     
sys.stdout = sys.__stdout__ 