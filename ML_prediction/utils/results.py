from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, balanced_accuracy_score
from utils import csvWriter
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
import seaborn as sns

def make_plot_file_name(model_name, configuration, kfold, fold_id):

    kFold = "_kFold" if kfold else ""
    fold_id = f"_fold_{fold_id}" if kfold else ""    
    lowVarianceDrop = "_lowVarianceDrop" if configuration[1] else ""
    dropExecTime = "_dropExecTime" if configuration[2] else ""
    rfecv = "_rfecv" if configuration[3] else ""
    kbest = "_kbest" if configuration[4] else ""
    autospearman = "_autospearman" if configuration[5] else ""
    hyperparameters = "_hyperparameters" if configuration[6] else ""
    features = f"_featureCount_{configuration[7]}"
    
    fileName = f"{model_name}{kFold}{fold_id}{lowVarianceDrop}{dropExecTime}{rfecv}{kbest}{autospearman}{hyperparameters}{features}"

    # Build config directory name (skip fold/model/feature info)
    config_parts = [
        "lowVarianceDrop" if configuration[1] else "",
        "noExecTime" if configuration[2] else "",
        "RFECV" if configuration[3] else "",
        f"KBest_{configuration[7]}" if configuration[4] else "",
        "autoSpearman" if configuration[5] else "",
        "hyperparameters" if configuration[6] else "",
    ]
    config_suffix = "_".join([p for p in config_parts if p]) or "default"
    config_dir = f"{'kFold' if kfold else 'simple'}_{config_suffix}"


    return fileName, config_dir

def plot_violin(y_test, y_predictions, model_name, configuration, kfold, fold_id, r2=None, mape=None, logScale=True):
    try:
        # === Get output paths ===
        root_dir = Path.cwd()
        fileName, config_dir = make_plot_file_name(model_name, configuration, kfold, fold_id)

        sub_dir = "kFold" if kfold else "simple"
        plots_dir = root_dir / "plots" / sub_dir / config_dir
        plots_dir.mkdir(parents=True, exist_ok=True)

        scale = "log" if logScale else "exponential"
        plot_file_path = plots_dir / f"{fileName}_violin_{scale}.png"

        # === Convert to 1D pandas Series ===
        y_test_series = pd.Series(y_test, name="Actual").astype(float)
        y_pred_series = pd.Series(y_predictions, name="Predicted").astype(float)

        print("Sample y_test:", y_test_series.head())
        print("Sample y_predictions:", y_pred_series.head())

        if not logScale:
            y_test_series = np.exp(y_test_series)
            y_pred_series = np.exp(y_pred_series)

        # === Basic sanity checks ===
        if y_test_series.shape[0] != y_pred_series.shape[0]:
            raise ValueError(f"[ERROR] Length mismatch: y_test has {len(y_test_series)}, y_predictions has {len(y_pred_series)}.")

        if y_test_series.isna().any() or y_pred_series.isna().any():
            raise ValueError("[ERROR] NaN values found in y_test or y_predictions.")

        if np.isinf(y_test_series).any() or np.isinf(y_pred_series).any():
            raise ValueError("[ERROR] Inf values found in y_test or y_predictions.")

        # === Prepare DataFrame for violin plot ===
        df_plot = pd.concat([y_test_series, y_pred_series], axis=1)
        df_melted = df_plot.melt(var_name='Type', value_name='log(Energy)')

        # === Plotting ===
        plt.figure()
        sns.violinplot(x='Type', y='log(Energy)', data=df_melted)

        # Add metrics if provided
        if r2 is not None or mape is not None:
            metrics_text = ""
            if r2 is not None:
                metrics_text += f"R² = {r2:.3f}\n"
            if mape is not None:
                metrics_text += f"MAPE = {mape:.2f}%"
            plt.text(0.01, 0.95, metrics_text.strip(), transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

        # Final plot settings
        ylabel = 'log(Energy Consumption in joules)' if logScale else 'Energy Consumption in joules'
        plt.ylabel(ylabel)
        plt.grid(True, axis='y')
        plt.tight_layout()

        # Save plot
        plt.savefig(plot_file_path)
        print(f"[INFO] Violin plot saved successfully at: {plot_file_path}")
        plt.close()

    except Exception as e:
        print(f"[EXCEPTION] Error while plotting violin plot for {model_name}: {e}")


def plot_box(y_test, y_predictions, model_name, configuration, kfold, fold_id, r2=None, mape=None, logScale=True):
    try:
        # === Output paths ===
        root_dir = Path.cwd()
        fileName, config_dir = make_plot_file_name(model_name, configuration, kfold, fold_id)
        sub_dir = "kFold" if kfold else "simple"
        plots_dir = root_dir / "plots" / sub_dir / config_dir
        plots_dir.mkdir(parents=True, exist_ok=True)

        scale = "log" if logScale else "exponential"
        plot_file_path = plots_dir / f"{fileName}_box_{scale}.png"

        # === Convert to pandas Series for safety ===
        y_test = pd.Series(y_test).astype(float)
        y_predictions = pd.Series(y_predictions).astype(float)

        # === Apply exponential if needed ===
        if not logScale:
            y_test = np.exp(y_test)
            y_predictions = np.exp(y_predictions)
        
        # === Validity checks ===
        if len(y_test) != len(y_predictions):
            raise ValueError(f"[ERROR] Length mismatch: y_test={len(y_test)}, y_predictions={len(y_predictions)}")

        if y_test.isna().any() or y_predictions.isna().any():
            raise ValueError("[ERROR] NaN values found.")
        if np.isinf(y_test.values).any() or np.isinf(y_predictions.values).any():
            raise ValueError("[ERROR] Inf values found.")

        # === Create DataFrame for plotting ===
        df_plot = pd.concat([y_test.rename("Actual"), y_predictions.rename("Predicted")], axis=1)
        df_melted = df_plot.melt(var_name="Type", value_name="Energy (Joules)")
        df_melted["Energy (Joules)"] = df_melted["Energy (Joules)"].astype(float)

        # === Plot ===
        plt.figure()
        sns.boxplot(x="Type", y="Energy (Joules)", data=df_melted, width=0.4)

        # === Add metrics ===
        if r2 is not None or mape is not None:
            metrics_text = ""
            if r2 is not None:
                metrics_text += f"R² = {r2:.3f}\n"
            if mape is not None:
                metrics_text += f"MAPE = {mape:.2f}%"
            plt.text(0.01, 0.95, metrics_text.strip(), transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

        # === Labels and save ===
        ylabel = "log(Energy Consumption in joules)" if logScale else "Energy Consumption in joules"
        plt.ylabel(ylabel)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(plot_file_path)
        print(f"[INFO] Box plot saved successfully at: {plot_file_path}")
        plt.close()

    except Exception as e:
        print(f"[EXCEPTION] Error while plotting box plot for {model_name}: {e}")



def plot_line(y_test, y_predictions, model_name, configuration, kfold, fold_id, r2=None, mape=None, logScale=True):

    # === Output paths ===
    root_dir = Path.cwd()
    fileName, config_dir = make_plot_file_name(model_name, configuration, kfold, fold_id)
    sub_dir = "kFold" if kfold else "simple"
    plots_dir = root_dir / "plots" / sub_dir / config_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    scale = "log" if logScale else "exponential"
    plot_file_path = plots_dir / f"{fileName}_line_{scale}.png"

    # === Convert to pandas Series ===
    y_test = pd.Series(y_test).astype(float)
    y_predictions = pd.Series(y_predictions).astype(float)

    if not logScale:
        y_test = np.exp(y_test)
        y_predictions = np.exp(y_predictions)


    plt.figure()
    plt.plot(y_test.values, color='red', label='Actual')
    plt.plot(y_predictions.values, color='green', label='Predicted')

    # === Add metrics text ===
    if r2 is not None or mape is not None:
        metrics_text = ""
        if r2 is not None:
            metrics_text += f"R² = {r2:.3f}\n"
        if mape is not None:
            metrics_text += f"MAPE = {mape:.2f}%"
        plt.text(0.01, 0.95, metrics_text.strip(), transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    ylabel = 'log(Energy Consumption in joules)' if logScale else 'Energy Consumption in joules'
    plt.xlabel('Test methods')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # === Highlight close predictions ===
    epsilon = 0.2
    match_indices = np.where(np.abs(y_predictions.values - y_test.values) < epsilon)[0]
    if len(match_indices) > 0:
        plt.scatter(
            match_indices,
            y_test.values[match_indices],
            color='blue',
            marker='o',
            s=30,
            label='Prediction ≈ Actual (<0.2)',
            zorder=5
        )
    plt.legend()
    plt.savefig(plot_file_path)
    plt.close()


def plot_scatter(y_test, y_predictions, model_name, configuration, kfold, fold_id, r2=None, mape=None, logScale=True):

    # === Output path setup ===
    root_dir = Path.cwd()
    fileName, config_dir = make_plot_file_name(model_name, configuration, kfold, fold_id)
    sub_dir = "kFold" if kfold else "simple"
    plots_dir = root_dir / "plots" / sub_dir / config_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    scale = "log" if logScale else "exponential"
    plot_file_path = plots_dir / f"{fileName}_scatter_{scale}.png"

    # === Ensure proper data types ===
    y_test = pd.Series(y_test).astype(float)
    y_predictions = pd.Series(y_predictions).astype(float)

    if not logScale:
        y_test = pd.Series(np.exp(y_test)).astype(float)
        y_predictions = pd.Series(np.exp(y_predictions)).astype(float)

    # === Plotting ===
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_predictions, color='mediumseagreen', label='Predicted', alpha=0.7)

    # Diagonal reference line
    min_val = min(y_test.min(), y_predictions.min())
    max_val = max(y_test.max(), y_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (Actual = Predicted)')

    # Metrics box
    if r2 is not None or mape is not None:
        metrics_text = ""
        if r2 is not None:
            metrics_text += f"R² = {r2:.3f}\n"
        if mape is not None:
            metrics_text += f"MAPE = {mape:.2f}%"
        plt.text(0.01, 0.95, metrics_text.strip(), transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    ylabel = 'log(Energy Consumption in joules)' if logScale else 'Energy Consumption in joules'
    plt.xlabel(f'Actual {ylabel}')
    plt.ylabel(f'Predicted {ylabel}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_file_path)
    plt.close()

def display(y_test, y_predictions, y_predictions_proba, model_name, fileName, configuration, sMode, task_type, kfold, fold_id):
    
    # === Output paths ===
    root_dir = Path.cwd()
    fileName_predictions, config_dir = make_plot_file_name(model_name, configuration, kfold, fold_id)
    sub_dir = "kFold" if kfold else "simple"
    plots_dir = root_dir / "plots" / sub_dir / config_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    # save actual and predicted values to CSV
    results_file_path = plots_dir / f"{fileName_predictions}_values.csv"
    pd.DataFrame({'Actual': y_test, 'Predicted': y_predictions}).to_csv(results_file_path, index=False)
    print(f"[INFO] Predictions and actuals saved at: {results_file_path}")

    kfoldConfig = [kfold, fold_id] if kfold else []

    if task_type == 'classification':
        y_predictions_proba = y_predictions_proba[:, 1]
        TN, FP, FN, TP = confusion_matrix(y_test, y_predictions).ravel()

        bal_acc = balanced_accuracy_score(y_test, y_predictions)
        acc = accuracy_score(y_test, y_predictions)
        auc = roc_auc_score(y_test, y_predictions_proba)
        mcc = matthews_corrcoef(y_test, y_predictions)
        precision = precision_score(y_test, y_predictions, average='binary')
        recall = recall_score(y_test, y_predictions, average='binary')
        f1 = f1_score(y_test, y_predictions, average='binary')

        resultsArray = kfoldConfig + configuration + [sMode,model_name,precision,recall,f1,acc,bal_acc,auc,mcc,TP,FP,TN,FN]    
    
    elif task_type == 'regression':
        mse = mean_squared_error(y_test, y_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_predictions)
        r2 = r2_score(y_test, y_predictions)
        mape = mean_absolute_percentage_error(y_test, y_predictions)

        resultsArray = kfoldConfig + configuration + [sMode, model_name, mse, rmse, mae, mape, r2]
        print(f"Results for {model_name} - R2: {r2}")
        
        try:

            plot_line(y_test, y_predictions, model_name, configuration, kfold, fold_id, r2=r2, mape=mape, logScale=True)
            plot_line(y_test, y_predictions, model_name, configuration, kfold, fold_id, r2=r2, mape=mape, logScale=False)
            
            plot_scatter(y_test, y_predictions, model_name, configuration, kfold, fold_id, r2=r2, mape=mape, logScale=True)
            plot_scatter(y_test, y_predictions, model_name, configuration, kfold, fold_id, r2=r2, mape=mape, logScale=False)

            plot_violin(y_test, y_predictions, model_name, configuration, kfold, fold_id, r2=r2, mape=mape, logScale=True)
            plot_violin(y_test, y_predictions, model_name, configuration, kfold, fold_id, r2=r2, mape=mape, logScale=False)

            plot_box(y_test, y_predictions, model_name, configuration, kfold, fold_id, r2=r2, mape=mape, logScale=True)
            plot_box(y_test, y_predictions, model_name, configuration, kfold, fold_id, r2=r2, mape=mape, logScale=False)
        except Exception as e: 
            # PRINT FULL ERROR MESSAGE WITH ALL DETALS FOR DEBUGGING
            print(f"Error while plotting violin plot for {model_name}: {e}")
    
    csvWriter.write(resultsArray, fileName, kFold = kfold, task_type=task_type)

def display_hyper_results(y_test, y_predictions, y_predictions_proba, model_name, fileName, configuration, sMode, task_type):
    
    if task_type == 'regression':
        mse = mean_squared_error(y_test, y_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_predictions)
        r2 = r2_score(y_test, y_predictions)
        mape = mean_absolute_percentage_error(y_test, y_predictions)

        resultsArray = configuration + [sMode, model_name, mse, rmse, mae, mape, r2]
        print(f"Results for {model_name} - R2: {r2}")
        plot_line(y_test, y_predictions, model_name, configuration, False, None, r2=r2, mape=mape)
        plot_scatter(y_test, y_predictions, model_name, configuration, False, None, r2=r2, mape=mape)

    csvWriter.write(resultsArray, fileName, kFold = False, task_type=task_type)