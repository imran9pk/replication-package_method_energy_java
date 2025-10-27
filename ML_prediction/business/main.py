from models import neuralNet, decisionTree, linearRegression, gradientBoosting, randomForest, supportVectorMachines, kneighbor, adaBoost, lassoRegression, ridgeRegression
import pandas as pd
import numpy as np

from utils import shapCalculator
from sklearn.feature_selection import VarianceThreshold

from pathlib import Path

root = Path.cwd()
data_dir = root / 'data'
energy_dataset = data_dir / "method_dataset_noIter.csv"
perf_dataset = data_dir / "metrics_combined_all-05-12-2023.csv"
useLog = True  # Whether to use log values for energy and execution time
PROFILING_ITERATIONS = 1

# Method to trop columns that we do not need in the dataset
def drop_cols(df, cols):
    for col in cols:
            if col in df.columns:
                df = df.drop(columns=col)
    return df

def get_low_variance_features(df, threshold):
    # Only keep numeric features
    numeric_df = df.select_dtypes(include=[np.number])

    # Apply VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)  # drops features with variance < 0.01
    selector.fit(numeric_df)

    # features_kept = numeric_df.columns[selector.get_support()]
    features_dropped = numeric_df.columns[~selector.get_support()]

    return features_dropped

def load_energy_dataset(path, lowVarianceDrop, dropExecTime):
    # Load dataset and rename columns
    df = pd.read_csv(path)
    df.rename(columns={
    'energy(joules)': 'energy',
    'execution_time(ms)': 'execution_time'
    }, inplace=True)

    # Drop duplicates and rows with NaN in 'energy' or 'execution_time'
    df.drop_duplicates(subset=['Method_name'], inplace=True)
    df.dropna(subset=['energy','execution_time'], inplace=True)

    # Drop rows where 'energy' or 'execution_time' is zero
    non_zero_df = df[df['energy'] > 0].copy()
    non_zero_df = non_zero_df[non_zero_df['execution_time'] > 0].copy()

    # divide the energy and execution times with profilingIterations
    non_zero_df['energy'] = non_zero_df['energy'] / PROFILING_ITERATIONS
    non_zero_df['execution_time'] = non_zero_df['execution_time'] / PROFILING_ITERATIONS

    # convert log values to positive values
    non_zero_df['log_energy'] = np.log(non_zero_df['energy']) 
    non_zero_df['log_execution_time'] = np.log(non_zero_df['execution_time'])

    # If useLog is False, keep original energy and execution_time values
    # However we still keep the name of cols as log_energy and log_execution_time for code consistency
    if not useLog:
        print("Using original energy and execution time values instead of log values.")
        non_zero_df['log_energy'] = non_zero_df['energy'] 
        non_zero_df['log_execution_time'] = non_zero_df['execution_time']

    # Convert 'methodScope' to categorical and one-hot encode it and covert to 
    non_zero_df = pd.get_dummies(non_zero_df, columns=["methodScope"], drop_first=True)
    bool_cols = non_zero_df.select_dtypes(include='bool').columns
    non_zero_df[bool_cols] = non_zero_df[bool_cols].astype('int64')

    # Drop columns that are not needed
    extra_cols = ['url', 'N_small', 'N_medium', 'N_large', 'command_line',
                 'methodCallNames', 'internalCallsList', 'externalCallsList',
                 'task', 'Class_Names', 'Method_name', 'methodType', 'isOverloaded', 'energy', 'execution_time']
    df_clean = drop_cols(non_zero_df, extra_cols)
    
    if dropExecTime:
        df_clean = df_clean.drop(columns=['log_execution_time'])
        
    low_variance_features = []
    if lowVarianceDrop:
        low_variance_features = get_low_variance_features(df_clean, threshold=0.01)
        print(f"Low Variance Features Dropped: {low_variance_features}")
        # low_variance_features = ['#case', 'usesJavaLangThread', 'usesJavaNio', 'usesJavaNioChannels', 'usesJavaNioFile', 'usesJavaNioCharset', 'usesJavaNet', 'usesJavaxNetSsl', 'usesJavaLangManagement', 'usesJavaUtilRegex', 'usesJavaText', 'methodScope_protected']
    
    df_clean = drop_cols(df_clean, low_variance_features)
    
    X = df_clean.drop(columns=["log_energy"]) 
    y = df_clean["log_energy"]

    # print the cols that are in X
    print(f"Features Loaded in X: {X.columns.tolist()}")
    return X, y, df_clean

def dataLoadingPandas(columns, lowVarianceDrop, dropExecTime, task_type):

    dataset = energy_dataset if task_type == "regression" else perf_dataset
    dataset_name = dataset.name

    print(f"Loading dataset at: {dataset} with name {dataset_name}")

    if task_type == "regression":

        X_Data, Y_Data, df_clean = load_energy_dataset(energy_dataset, lowVarianceDrop, dropExecTime)
        default_cols = df_clean.columns.tolist()

        if len(columns) > 0:
            # Filter columns based on the provided list
            columns = [col for col in columns if col in default_cols]
            df_clean = df_clean[columns + ['log_energy']]
            X_Data = X_Data[columns]

        # print("X Count" + str(X_Data.count()))
        # print("Composition: ")
        # print("y Count" + str(Y_Data.count()))
        # print("y Value Counts: " + str(Y_Data.value_counts()))
        # print("y Unique: " + str(Y_Data.unique()))

        # loaded_dataset_name = dataset.split("/")[-1]
        # return X_Data, Y_Data, df_clean, loaded_dataset_name

    elif task_type == "classification":
        
        if(dataset_name == 'metrics_combined_all-05-12-2023.csv'):
            columns = ["methodScope", "isOverloaded", "methodLoc", "#for", "#while", "#do", "#nestedLoops", "#if", "#switch", "#case", "#return", "#throw", "#catch", "cyclo", "#vars", "#methodCalls",
                "#internalCalls", "#externalCalls", "#cPkgClasses", "cIsAbstract", "cNestingLevel", "#cImports", "#cNativeImports", "classLOC", "#cMethods", "#cInherits", "#cImplements", "usesConcurrency", "usesCollection", "isBenchmarked"]

        else:
            columns_default = ["methodScope","nameLen", "isOverloaded", "methodLoc", "#for", "#while", "#do", "#nestedLoops", "#if", "#switch", "#case", "#return", "#throw", "#catch", "cyclo", "#vars", "#methodCalls",
                    "#internalCalls", "#externalCalls", "usesJavaUtil", "usesJavaLangThread", "usesJavaUtilConcurrent", "usesJavaIo", "usesJavaNio", "usesJavaNioChannels", "usesJavaNioFile", "usesJavaNioCharset" ,"usesJavaNet",
            "usesJavaxNetSsl", "usesJavaLang", "usesJavaLangManagement", "usesJavaUtilRegex", "usesJavaText", "usesJavaMath", "#cPkgClasses","classScope", "cIsAbstract", "cNestingLevel", "#cImports", "#cNativeImports", "classLOC", "#cMethods", "#cInherits", "#cImplements", "isBenchmarked"]

        if(len(columns)>0):
            columns = np.append(columns, "isBenchmarked")#columns.append(["isBenchmarked"])
            print(columns)
            Scopes = {'public': '1', 'protected': '2', 'private': '3', 'default': '4'}
            df = pd.read_csv(dataset, usecols=columns)
            if('methodScope' in columns):
                df['methodScope'] = df['methodScope'].map(Scopes)
            if('classScope' in columns):
                df['classScope'] = df['classScope'].map(Scopes)

        else:
            Scopes = {'public': '1', 'protected': '2', 'private': '3', 'default': '4'}
            df = pd.read_csv(dataset, usecols=columns_default)
            df['methodScope'] = df['methodScope'].map(Scopes)
            df['classScope'] = df['classScope'].map(Scopes)

        Y_Data = df['isBenchmarked']
        X_Data = df.drop(columns='isBenchmarked')

        # print(X_Data.count())
        # print("Composition: ")
        # print(Y_Data.count())
        # print(Y_Data.value_counts())
        # print(Y_Data.unique())

        # loaded_dataset_name = dataset.split("/")[-1]
        # # return X_Data, Y_Data, df, loaded_dataset_name
    
    print("X Count" + str(X_Data.count()))
    print("Composition: ")
    print("y Count" + str(Y_Data.count()))
    print("y Value Counts: " + str(Y_Data.value_counts()))
    print("y Unique: " + str(Y_Data.unique()))

    return X_Data, Y_Data, df_clean, dataset_name

def predict(model, hyperparameters, X_train, X_test, y_train, y_test, random_state, SHAP, configuration, kfold, fold_id, task_type):

    print("#############")
    print("#############")
    print("TESTING " + model)
    print("#############")
    print("#############")


    if(model=="MLP"):
        # Multi Layer Perceptron
        y_predictions, y_predictions_proba, classifier = neuralNet.net(X_train, X_test, y_train, random_state, task_type)

    if(model=="DT"):

        if (hyperparameters):
            criterion = "entropy"
            max_depth = None
            max_features = None
            min_samples_leaf = 1
        else:
            criterion = None
            max_depth = None
            max_features = None
            min_samples_leaf = None
        y_predictions, y_predictions_proba, classifier = decisionTree.DT(hyperparameters, X_train, X_test, y_train, criterion, max_depth, max_features, min_samples_leaf, random_state, task_type)

    if(model=="LR"):
        y_predictions, y_predictions_proba, classifier = linearRegression.LR(X_train, X_test, y_train, random_state, task_type)

    if(model=="GB"):
        y_predictions, y_predictions_proba, classifier = gradientBoosting.GB(X_train, X_test, y_train, random_state, task_type)

    if(model=="HGB"):
        y_predictions, y_predictions_proba, classifier = gradientBoosting.HGB(X_train, X_test, y_train, random_state, task_type)

    if(model=="RF"):
    # Random Forest
    #
        if(hyperparameters):
            max_depth = None
            max_features = 'sqrt'
            min_sample_leaf = 1
            n_estimators = 2000
        else:
            max_depth = None
            max_features = 'sqrt'
            min_sample_leaf = None
            n_estimators = None

        y_predictions, y_predictions_proba, classifier = randomForest.rf(hyperparameters, X_train, X_test, y_train, max_depth, n_estimators, max_features, min_sample_leaf, random_state, task_type)

    if(model=="SVM"):
    # Support Vector Machines
    #
        kernel = "linear" #rbf
        y_predictions, y_predictions_proba, classifier = supportVectorMachines.SVM(X_train, X_test, y_train, kernel, task_type)

    if(model=="kNN"):
    # k-nearest neighbor
    #
        if(hyperparameters):
            algorithm = "kd_tree"
            leaf_size = 35
            metric = "minkowski"
            n_neighbors = 1
            weights = "distance"
        else:
            algorithm = None
            leaf_size = None
            metric = None
            n_neighbors = None
            weights = None

        y_predictions, y_predictions_proba, classifier = kneighbor.kNN(hyperparameters, X_train, X_test, y_train, algorithm, leaf_size, metric, n_neighbors, weights, task_type)

    if(model=="ADA"):
    # ADABoost
    #
        y_predictions, y_predictions_proba, classifier = adaBoost.ada(hyperparameters, X_train, X_test, y_train, random_state, task_type)
    
    if(model=="LassoR"):
        y_predictions, y_predictions_proba, classifier = lassoRegression.LassoR(X_train, X_test, y_train, random_state, task_type)

    if(model=="RidgeR"):
        y_predictions, y_predictions_proba, classifier = ridgeRegression.RidgeR(X_train, X_test, y_train, random_state, task_type)

    if(SHAP):
        shapCalculator.calculateShap(classifier, X_test, task_type, configuration, kfold, fold_id, model_name=model)

    return y_test, y_predictions, y_predictions_proba, model