# %%
import pandas as pd
from pathlib import Path

# %%
def generate_commands(df, profiler_base, joular_agent, repeat, output_dir, clbg, rosetta):

    output_dir = Path(output_dir)
    
    # Generate Commands
    commands = []

    for _, row in df.iterrows():

        class_name = row["task"] if clbg else row["class_name"]
        
        #rosetta does not have varied input sizes with "N_small", "N_medium", "N_large"
        input_values_list = [0]
        if clbg:
            input_values_list = [row["N_small"], row["N_medium"], row["N_large"]]
            
        for input_value in input_values_list:
            jfr_file = f"{output_dir}/runtime_{class_name}_{input_value}.text"

            command = (
                f"java -agentpath:{profiler_base}=start,event=cpu,interval=100us,file={jfr_file} "
                f"-javaagent:{joular_agent} "
                f"-cp . LoopRunner {repeat} {class_name} {input_value}"
            )

            commands.append({
                "task": class_name,
                "input_value": input_value,
                "command": command
            })

    return commands

# %%
if __name__ == "__main__":
    # Configuration
    root_dir = Path.cwd().parent
    data_dir = root_dir / "data"
    output_dir = root_dir / "outputs"

    input_csv_clbg = data_dir / "clbg_problems_data.csv"
    input_csv_rosetta = data_dir / "rosetta_problems_data.csv"
    commands_output_csv = data_dir / "LoopRunner_commands2.csv"

    repeat = 20
    profiler_base = "/home/imran/tools/async-profiler/lib/libasyncProfiler.so"
    joular_agent = "/home/imran/tool_tests/joularjx/joularjx-3.0.1.jar"

    clbg = True
    rosetta = False

    csv = input_csv_clbg if clbg else input_csv_rosetta
    df = pd.read_csv(csv)

    if "included" in df.columns:
        df = df[df["included"] != False]  # Exclude rows explicitly marked False
    elif "dropped" in df.columns:
        df = df[df["dropped"] != True]

    # Generate Commands
    commands = generate_commands(df, profiler_base, joular_agent, repeat, output_dir, clbg, rosetta)

    # Export
    commands_df = pd.DataFrame(commands)
    commands_df.drop_duplicates(subset=["task"], inplace=True)

    commands_df.to_csv(commands_output_csv, index=False)
    print(f"Saved {len(commands)} commands to {commands_output_csv}")



