import pandas as pd
from sklearn.model_selection import train_test_split

# Features
X = process_data.drop('class', axis=1)

# Target
y = process_data['class']

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,   # 20% for testing
    random_state=42, # for reproducibility
    stratify=y       # maintain class distribution
)

# Check shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)