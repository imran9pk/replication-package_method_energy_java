import pandas as pd


def RQ1(nome_file_csv):#precision/recall/balanced_accuracy/f1score plain version
    # Leggi il file CSV in un DataFrame
    # names=['modello', 'f1_score', 'kfold']
    try:
        df = pd.read_csv(nome_file_csv, header=None,
                         names=["dataset_name", "rfecv", "autospearman", "hyperparameters", "features", "sampling",
                                "model_name", "precision", "recall", "f1_score", "accuracy", "balanced_accuracy", "auc",
                                "mcc", "TP", "FP", "TN", "FN"])
    except FileNotFoundError:
        print(f"Error: The file '{nome_file_csv}' was not found.")
        exit()

    # Raggruppa per nome del modello e calcola la media dell'f1_score
    metrics = ['f1_score','precision','recall','balanced_accuracy']
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
        #risultato = df.groupby(['model_name', 'sampling', 'rfecv', 'autospearman'])[metric].mean().reset_index()
        risultato = df.groupby(['model_name','sampling'])[metric].mean().reset_index()
        risultato = risultato.loc[risultato['sampling']=='Simple']
        #df.loc[df['column_name'] == some_value]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(risultato)

def RQ2(nome_file_csv):#precision/recall/balanced_accuracy/f1score plain version
    # Leggi il file CSV in un DataFrame
    # names=['modello', 'f1_score', 'kfold']
    try:
        df = pd.read_csv(nome_file_csv, header=None,
                         names=["dataset_name", "rfecv", "autospearman", "hyperparameters", "features", "sampling",
                                "model_name", "precision", "recall", "f1_score", "accuracy", "balanced_accuracy", "auc",
                                "mcc", "TP", "FP", "TN", "FN"])
    except FileNotFoundError:
        print(f"Error: The file '{nome_file_csv}' was not found.")
        exit()

    # Raggruppa per nome del modello e calcola la media dell'f1_score
    metric='f1_score'
    samplings=['Simple','Over','Under']
    for sampling in samplings:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
        risultato = df.groupby(['model_name', 'sampling', 'rfecv', 'autospearman',''])[metric].mean().reset_index()
        #risultato = df.groupby(['model_name','sampling',])[metric].mean().reset_index()
        risultato = risultato.loc[risultato['sampling']==sampling]
        #df.loc[df['column_name'] == some_value]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(risultato)

def RQ3(nome_file_csv):#precision/recall/balanced_accuracy/f1score plain version
    # Leggi il file CSV in un DataFrame
    # names=['modello', 'f1_score', 'kfold']
    try:
        df = pd.read_csv(nome_file_csv, header=None,
                         names=["dataset_name", "rfecv", "autospearman", "hyperparameters", "features", "sampling",
                                "model_name", "precision", "recall", "f1_score", "accuracy", "balanced_accuracy", "auc",
                                "mcc", "TP", "FP", "TN", "FN"])
    except FileNotFoundError:
        print(f"Error: The file '{nome_file_csv}' was not found.")
        exit()

    # Raggruppa per nome del modello e calcola la media dell'f1_score
    metrics = ['f1_score','precision','recall','balanced_accuracy']
    #metrics = ['precision']
    #samplings = ['Simple', 'Over', 'Under']
    samplings = ['Simple']
    metric = 'balanced_accuracy'
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    #risultato = df.groupby(['model_name', 'sampling','hyperparameters'])[metric].mean().reset_index()
    risultato = df.groupby(['model_name','sampling','rfecv','autospearman','hyperparameters'])[metric].mean().reset_index()
    #risultato = risultato.loc[risultato['sampling']==sampling]
    #df.loc[df['column_name'] == some_value]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #print(risultato.loc[risultato['model_name']=='RF'])
        risultato = risultato.loc[risultato['hyperparameters']=='True']
        risultato = risultato.loc[risultato['model_name'] == 'DT']
        print(risultato.sort_values(by=[metric]))
