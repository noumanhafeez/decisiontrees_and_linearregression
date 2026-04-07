import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------
# Load your processed data
# ------------------------------
process_data = pd.read_csv("../data/processed_mushrooms.csv")  # Adjust path if needed

# ------------------------------
# Split into features and target
# ------------------------------
target_column = 'class'  # Change if your target has a different name
X = process_data.drop(target_column, axis=1)  # All columns except target
y = process_data[target_column]               # Target column

# ------------------------------
# Split into train and test sets
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% test, 80% train
    random_state=42,     # for reproducibility
    stratify=y           # preserve class distribution
)

# ------------------------------
# Print shapes
# ------------------------------
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)