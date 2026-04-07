import os
import kagglehub
import shutil

# ------------------------------
# Step 1: Set target folder
# ------------------------------
target_dir = os.path.join("..", "data")  # decision_tree/data relative to src
os.makedirs(target_dir, exist_ok=True)

# ------------------------------
# Step 2: Download dataset (cached location)
# ------------------------------
dataset_path = kagglehub.dataset_download("uciml/mushroom-classification")
print("Downloaded dataset path (cache):", dataset_path)

# ------------------------------
# Step 3: Copy dataset to target folder
# ------------------------------
shutil.copytree(dataset_path, target_dir, dirs_exist_ok=True)
print("Dataset successfully saved to:", target_dir)