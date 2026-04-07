import pandas as pd
import math

# ------------------------------
# Load processed dataset
# ------------------------------
df = pd.read_csv("../data/processed_mushrooms.csv")  # Already encoded or numeric

# For demonstration, assume target column is 'class'
target = 'class'


# ------------------------------
# Helper Functions
# ------------------------------

def entropy(data, target_col):
    """
    Calculate entropy of a dataset
    """
    values = data[target_col].value_counts()
    total = len(data)
    ent = 0
    for count in values:
        p = count / total
        ent -= p * math.log2(p)
    return ent


def information_gain(data, feature, target_col):
    """
    Calculate information gain of a feature
    """
    total_entropy = entropy(data, target_col)
    vals = data[feature].unique()
    weighted_entropy = 0
    for val in vals:
        subset = data[data[feature] == val]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset, target_col)
    return total_entropy - weighted_entropy


# ------------------------------
# ID3 Algorithm
# ------------------------------

def id3(data, target_col, features):
    """
    Recursively build ID3 tree
    """
    # If all examples have same class, return that class
    if len(data[target_col].unique()) == 1:
        return data[target_col].iloc[0]

    # If no features left, return majority class
    if len(features) == 0:
        return data[target_col].mode()[0]

    # Choose the best feature based on information gain
    gains = {feature: information_gain(data, feature, target_col) for feature in features}
    best_feature = max(gains, key=gains.get)

    tree = {best_feature: {}}

    # Recur for each value of best feature
    for val in data[best_feature].unique():
        subset = data[data[best_feature] == val]
        # Remove the feature from list for recursion
        new_features = [f for f in features if f != best_feature]
        tree[best_feature][val] = id3(subset, target_col, new_features)

    return tree


# ------------------------------
# Build Tree
# ------------------------------

features = list(df.columns)
features.remove(target)

decision_tree = id3(df, target, features)

# ------------------------------
# Print Tree
# ------------------------------
import pprint

pprint.pprint(decision_tree)