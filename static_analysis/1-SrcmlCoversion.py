# %%
import os
import subprocess
from pathlib import Path

def convert_java_to_srcml(source_dir, target_dir, overwrite=False):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.endswith(".java"):
            java_path = source_dir / filename 
            base_name = java_path.stem 
            xml_path = target_dir / f"{base_name}.xml"

            if xml_path.exists() and not overwrite:
                print(f"Skipping (already exists): {xml_path}")
                continue

            try:
                with open(xml_path, 'w+', encoding='utf-8') as f:
                    subprocess.run(['srcml', java_path], stdout=f, check=True)
                print(f"Converted: {filename} → {xml_path}")
            except subprocess.CalledProcessError:
                print(f"Failed to convert: {filename}")

# %%
if __name__ == "__main__":
    root_dir = Path.cwd().parent
    source_dir_rosetta = root_dir / "data" / "rosetta_code"
    target_dir_rosetta = root_dir / "data" / "srcml_rosetta"

    source_dir_clbg = root_dir / "data" / "clbg_code"
    target_dir_clbg = root_dir / "data" / "srcml_clbg"
    
    convert_java_to_srcml(source_dir_rosetta,target_dir_rosetta,overwrite=False)


