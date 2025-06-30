import os
import pandas as pd

def log_diagnostics(y_true, y_pred, model_name, output_csv, fold_id=None):

    df = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
    })
    df["residual"] = df["actual"] - df["predicted"]
    df["percent_error"] = 100 * abs(df["residual"]) / df["actual"].replace(0, 1e-9)
    df["model"] = model_name
    df["fold"] = fold_id if fold_id is not None else -1

    # Append to file (with header if it doesn't exist)
    write_header = not os.path.exists(output_csv)
    df.to_csv(output_csv, mode='a', index=False, header=write_header)
    print(f"Logged diagnostics to: {output_csv}")
