from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error
from utils import csvWriter



def display(y_test, y_pred, model_name, fileName, configuration, sMode):
    """
    Logs regression performance metrics to file.

    Parameters:
        y_test (pd.Series): True energy values.
        y_pred (np.array): Predicted energy values.
        model_name (str): Name of the model used.
        fileName (str): Path to save results CSV.
        configuration (list): Experiment config info.
        sMode (str): Sampling mode used.
    """

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    resultsArray = configuration + [sMode, model_name, mae, rmse, r2]

    csvWriter.write(resultsArray, fileName, kFold=False)
