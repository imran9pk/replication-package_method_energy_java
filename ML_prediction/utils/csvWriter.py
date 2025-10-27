import csv

def write(results,fileName,kFold, task_type):

    with open(fileName, 'a+', newline='') as csvfile:
        results_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csvfile.seek(0)
        line = csvfile.readline()
        kFold_regression_header = ["kFold","kFoldID","dataset_name","lowVarianceDrop","dropExecTime","rfecv","kbest","autospearman","hyperparameters","features","sampling","model_name","mse","rmse","mae","mape","r2"]
        simple_regression_header = ["dataset_name","lowVarianceDrop","dropExecTime","rfecv","kbest","autospearman","hyperparameters","features","sampling","model_name","mse","rmse","mae","mape","r2"]
        kFold_classification_header = ["kFold","kFoldID","dataset_name","lowVarianceDrop","dropExecTime","rfecv","kbest","autospearman","hyperparameters","features","sampling","model_name","precision","recall","f1_score","accuracy","balanced_accuracy","auc","mcc","TP","FP","TN","FN"]
        simple_classification_header = ["dataset_name","lowVarianceDrop","dropExecTime","rfecv","kbest","autospearman","hyperparameters","features","sampling","model_name","precision","recall","f1_score","accuracy","balanced_accuracy","auc","mcc","TP","FP","TN","FN"]

        if task_type == "regression":
            header = kFold_regression_header if kFold else simple_regression_header
        else:
            header = kFold_classification_header if kFold else simple_classification_header
        
        if "model" not in line:
            results_writer.writerow(header)

        results_writer.writerow(results)


