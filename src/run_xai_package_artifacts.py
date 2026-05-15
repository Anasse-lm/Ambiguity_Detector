#!/usr/env/bin python3
import os
import zipfile
import datetime
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    xai_dir = root / "outputs/xai"
    schemas_dir = root / "schemas"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = root / f"outputs/xai_artifacts_{timestamp}.zip"
    
    if xai_dir.exists():
        with open(xai_dir / "README.md", "w") as f:
            f.write("# XAI Phase Artifacts\n\nGenerated automatically via thesis pipeline.")
            
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if xai_dir.exists():
            for folder, _, files in os.walk(xai_dir):
                for file in files:
                    filepath = Path(folder) / file
                    arcname = filepath.relative_to(root)
                    zf.write(filepath, arcname)
                    
        if schemas_dir.exists():
            for folder, _, files in os.walk(schemas_dir):
                for file in files:
                    filepath = Path(folder) / file
                    arcname = filepath.relative_to(root)
                    zf.write(filepath, arcname)
                
    print(f"Artifacts packaged successfully.")
    print(f"ZIP PATH: {zip_path.absolute()}")

if __name__ == "__main__":
    main()
