import pandas as pd

def load_energy_dataset(path):
    df = pd.read_csv(path)
    df_clean = df.dropna(subset=["energy(joules)", "execution_time(ms)"])

    drop_cols = ['url', 'N_small', 'N_medium', 'N_large', 'command_line',
                 'methodCallNames', 'internalCallsList', 'externalCallsList',
                 'clbg_problem', 'Class_Names', 'Method_name', 'methodType']
    df_clean = df_clean.drop(columns=drop_cols)

    df_encoded = pd.get_dummies(df_clean, columns=["methodScope"])
    X = df_encoded.drop(columns=["energy"]) 
    y = df_encoded["energy"]

    return X, y