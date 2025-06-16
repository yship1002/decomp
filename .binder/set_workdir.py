# .binder/set_workdir.py

import os
desired_dir = "/home/jovyan/target_dir"

try:
    os.chdir(desired_dir)
    print(f"[INFO] Changed working directory to: {desired_dir}")
except Exception as e:
    print(f"[WARNING] Failed to change working directory: {e}")
