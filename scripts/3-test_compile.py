# %%
import os
import subprocess
from pathlib import Path
import pandas as pd

# Remove all the class files
def clean_build_files(dir):
    files_dir = Path(dir)
    count = 0
    for class_file in files_dir.rglob("*.class"):
        try:
            class_file.unlink()
            count += 1
        except Exception as e:
            print(f"Failed to delete {class_file}: {e}")

    print(f"\nDeleted {count} .class files from {dir}")

def test_compile_run(java_dir, input_value):

    clean_build_files(java_dir)
    java_files = list(java_dir.glob("*.java"))
    print(f"Found {len(java_files)} Java files in {java_dir}")

    results = {}

    # Process each file
    # Compile & Optionally Run
    for java_file in java_files:
        class_name = java_file.stem
        print(f"\nProcessing: {java_file.name}")
        results[java_file.name] = {'compiled': False, 'executed': False}

        # Compile
        compile_result = subprocess.run(['javac', str(java_file)], capture_output=True, text=True)
        if compile_result.returncode == 0:
            print(f"Compiled successfully: {java_file.name}")
            results[java_file.name]['compiled'] = True
        else:
            print(f"Compilation failed: {java_file.name}\n{compile_result.stderr.strip()}")
            java_file.unlink()
            continue

        # # Run (optional)
        # try:
        #     run_result = subprocess.run(['java', class_name, input_value], capture_output=True, text=True, timeout=5)
        #     if run_result.returncode == 0:
        #         print(f"Executed successfully: {class_name}")
        #         print(run_result.stdout)
        #         results[java_file.name]['executed'] = True
        #     else:
        #         print(f"Execution failed: {class_name}\n{run_result.stderr.strip()}")
        # except subprocess.TimeoutExpired:
        #     print(f"Execution timed out for: {class_name}")





    # for java_file in java_files:
    #     class_name = os.path.splitext(java_file)[0]

    #     print(f"\nProcessing: {java_file}")

    #     results[java_file] = {'compiled': False, 'executed': False}

    #     # Step 1: Compile
    #     compile_result = subprocess.run(['javac', java_file], capture_output=True, text=True)
    #     if compile_result.returncode == 0:
    #         print(f"Compiled successfully: {java_file}")
    #         results[java_file]['compiled'] = True
    #     else:
    #         print(f"Compilation failed: {java_file}\n{compile_result.stderr.strip()}")
    #         #delete the Failed java file
    #         os.remove(java_file)  # Remove the failed Java file
    #         continue  # Skip execution if compilation failed

        # # Step 2: Run
        # try:
        #     run_result = subprocess.run(['java', class_name, input_value], capture_output=True, text=True, timeout=5)
        #     if run_result.returncode == 0:
        #         print(f"Executed successfully: {class_name}")
        #         print(run_result.stdout)
        #         results[java_file]['executed'] = True
        #     else:
        #         print(f"Execution failed: {class_name}\n{run_result.stderr.strip()}")
        # except subprocess.TimeoutExpired:
        #     print(f"Execution timed out for: {class_name}")
        #     results[java_file]['executed'] = False
    
    return results


# Main script execution
if __name__ == "__main__":
    # Configuration
    root_dir = Path.cwd().parent
    java_dir = root_dir / "data" / "rosetta_code_flat"
    compile_results_csv = root_dir / "data" / "code_results_compiles.csv"
    input_value = '5'

    results = test_compile_run(java_dir, input_value)

    # Summary report
    print("\nPER-FILE RESULTS")
    for file, status in results.items():
        print(f"{file} → Compiled: {'Success' if status['compiled'] else 'Fail'}, Executed: {'Success' if status['executed'] else 'Fail'}")

    #write results to a csv
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(compile_results_csv, index_label="file_name")


