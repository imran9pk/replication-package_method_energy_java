import csv
import os

def write(results, fileName, kFold=False):

    header = ["dataset_name", "rfecv", "autospearman", "hyperparameters",
              "features", "sampling", "model_name", "MAE", "RMSE", "R2"]

    file_exists = os.path.isfile(fileName)
    
    with open(fileName, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # If new file, write header first
        if not file_exists or os.stat(fileName).st_size == 0:
            writer.writerow(header)

        writer.writerow(results)
