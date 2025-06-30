import os
import subprocess
import xml.etree.ElementTree as ET
import csv
from pathlib import Path
import shutil

# Methods for Filtering based on srcML
def has_main_method(xml_root):
    return xml_root.find(".//src:function[src:name='main']", SRCML_NS) is not None
    
def uses_scanner(xml_root):
    return any("Scanner" in (e.text or "") for e in xml_root.findall(".//src:name", SRCML_NS))

def uses_gui_import(xml_root):
    gui_pkgs = ["java.awt", "javax.swing", "javafx"]
    for imp in xml_root.findall(".//src:import/src:name", SRCML_NS):
        full = "".join(e.text or "" for e in imp.iter())
        if any(pkg in full for pkg in gui_pkgs):
            return True
    return False

def uses_stdin_input(xml_root):
    # Look for System.in, BufferedReader, InputStreamReader, StreamTokenizer
    symbols = ["System.in", "BufferedReader", "InputStreamReader", "StreamTokenizer"]
    for name_elem in xml_root.findall(".//src:name", SRCML_NS):
        full_text = "".join(e.text or "" for e in name_elem.iter())
        if any(s in full_text for s in symbols):
            return True
    return False

def extract_public_class(xml_root):
    for tag in ['class', 'interface', 'enum']:
        elem = xml_root.find(f".//src:{tag}/src:name", SRCML_NS)
        if elem is not None:
            return elem.text
    return None
   
def has_infinite_while_loop(xml_root):
    for while_elem in xml_root.findall(".//src:while", SRCML_NS):
        # Is it while (true)?
        condition = while_elem.find("src:condition/src:expr/src:literal", SRCML_NS)
        if condition is None or condition.text != "true":
            continue

        # Get body of the loop
        body = while_elem.find("src:block", SRCML_NS)
        if body is None:
            return True  # no body, no break/return = infinite

        # Look for <break> or <return> inside the loop body
        has_exit = body.find(".//src:break", SRCML_NS) is not None or \
                   body.find(".//src:return", SRCML_NS) is not None

        if not has_exit:
            return True  # infinite loop with no break or return

    return False  # all while(true) loops had breaks or there were none

def has_infinite_for_loop(xml_root):
    for for_elem in xml_root.findall(".//src:for", SRCML_NS):
        control = for_elem.find("src:control", SRCML_NS)
        if control is None:
            continue

        condition = control.find("src:condition", SRCML_NS)
        if condition is None:
            continue

        # Case 1: Empty condition (for(;;))
        is_unbounded = len(condition) == 0

        # Case 2: Condition is explicitly 'true' (for(...; true; ...))
        literal = condition.find(".//src:literal", SRCML_NS)
        if literal is not None and literal.text == "true":
            is_unbounded = True

        if is_unbounded:
            # Now check for exit points (break or return)
            body = for_elem.find("src:block", SRCML_NS)
            if body is None:
                return True  # no body means infinite

            has_exit = (
                body.find(".//src:break", SRCML_NS) is not None or
                body.find(".//src:return", SRCML_NS) is not None
            )

            if not has_exit:
                return True  # infinite loop with no exit mechanism

    return False  # All 'for' loops are safe

def convert_to_srcml(java_path, xml_path):
    try:
        with open(xml_path, 'w', encoding="utf-8") as out:
            subprocess.run(["srcml", str(java_path)], stdout=out, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

# Main process starts here
def process_files(code_download_dir, log_csv):
    log_rows = []
    processed_count = 0
    deleted_count = 0
    passed_count = 0
    for java_path in code_download_dir.rglob("*.java"):
        processed_count += 1
        task_name = java_path.parent.name
        xml_path = java_path.with_suffix(".xml")

        if not convert_to_srcml(java_path, xml_path):
            reason = "srcML error"
            log_rows.append([str(java_path), task_name, True, reason, "", ""])
            os.remove(java_path)
            deleted_count += 1
            continue

        try:
            root = ET.parse(xml_path).getroot()

            # Filter checks
            if not has_main_method(root):
                reason = "No main"
            elif uses_scanner(root) or uses_stdin_input(root):
                reason = "Requires User Input"
            elif uses_gui_import(root):
                reason = "Uses GUI"
            elif has_infinite_while_loop(root):
                reason = "Infinite While Loop"
            elif has_infinite_for_loop(root):
                reason = "Infinite For Loop"
            else:
                reason = "OK"

            if reason != "OK":
                os.remove(java_path)
                deleted_count += 1
                print(f"Dropped: {java_path.name} → {reason}")
                log_rows.append([str(java_path), task_name, True, reason, "", ""])
                continue

            # Rename the class based on filename
            class_name = extract_public_class(root)
            if not class_name:
                reason = "No public class"
                os.remove(java_path)
                deleted_count += 1
                print(f"Dropped: {java_path.name} → {reason}")
                log_rows.append([str(java_path), task_name, True, reason, "", ""])
                continue

            new_path = java_path.parent / f"{class_name}.java"
            if java_path.name != new_path.name:
                os.rename(java_path, new_path)
                print(f"Renamed: {java_path.name} → {new_path.name}")
                passed_count += 1
            else:
                print(f"Kept: {java_path.name}")
                passed_count += 1
            
            log_rows.append([str(java_path), task_name, False, "OK", str(new_path.name), class_name])

        except Exception as e:
            print(f"Parse failed: {java_path} → {e}")
            os.remove(java_path)
            deleted_count += 1
            log_rows.append([str(java_path), task_name, True, f"ERROR: {e}", "", ""])
        finally:
            if os.path.exists(xml_path):
                os.remove(xml_path)

    # Write log CSV
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "task", "dropped", "reason", "final_filename", "class_name"])
        writer.writerows(log_rows)

    print("\nSUMMARY")
    print(f"Processed files: {processed_count}")
    print(f"Deleted files: {deleted_count}")
    print(f"Passed files: {passed_count}")
    print(f"\nLog saved to: {log_csv}")

def remove_empty_dirs(code_download_dir):
    for dirpath, dirnames, filenames in os.walk(code_download_dir, topdown=False):
        if not filenames and not dirnames:
            os.rmdir(dirpath)
            print(f"Removed empty directory: {dirpath}")

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

def copy_java_files_to_flat_dir(code_download_dir, target_dir):

    source_dir = Path(code_download_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    java_files = list(source_dir.rglob("*.java"))
    print(f"Found {len(java_files)} .java files in {source_dir.resolve()}")

    copied = 0
    skipped = 0

    for source_file in java_files:
        dest_file = target_dir / source_file.name

        if dest_file.exists():
            if source_file.stat().st_size > dest_file.stat().st_size:
                shutil.copy2(source_file, dest_file)
                print(f"Replaced with larger: {source_file.name}")
                copied += 1
            else:
                skipped += 1
        else:
            shutil.copy2(source_file, dest_file)
            print(f"Copied: {source_file.name}")
            copied += 1

    print(f"\nDone. Copied: {copied}, Skipped: {skipped}")
    print(f"Output in: {target_dir.resolve()}")


# Main script execution
if __name__ == "__main__":
    # Setting up
    SRCML_NS = {'src': 'http://www.srcML.org/srcML/src'}

    root_dir = Path.cwd().parent
    data_folder = root_dir / "data"

    OUT_CSV = data_folder / "rosetta_java_task_index.csv"
    code_download_dir = data_folder / "rosetta_code_grouped"
    flat_code_dir = data_folder / "rosetta_code_flat"

    # Ensure required folders exist
    os.makedirs(code_download_dir, exist_ok=True)
    os.makedirs(flat_code_dir, exist_ok=True)
    code_download_dir = root_dir / "data" / "rosetta_code_grouped"
    log_csv = root_dir / "data" / "rosetta_problems_data.csv"
    flat_code_dir = root_dir / "data" / "rosetta_code_flat"

    clean_build_files(code_download_dir)
    process_files(code_download_dir, log_csv)
    remove_empty_dirs(code_download_dir)
    copy_java_files_to_flat_dir(code_download_dir, flat_code_dir)
