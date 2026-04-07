import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------
# 1️⃣ Load dataset
# ------------------------------
df = pd.read_csv("../data/processed_data.csv")  # replace with your CSV path

# ------------------------------
# 2️⃣ Split features and target
# ------------------------------
target_column = 'target'  # replace with your target column name
X = df.drop(target_column, axis=1)
y = df[target_column]

# ------------------------------
# 3️⃣ Split into train and test sets
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ------------------------------
# 4️⃣ Feature scaling (optional for linear regression, but helps)
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 5️⃣ Train Linear Regression model
# ------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ------------------------------
# 6️⃣ Make predictions
# ------------------------------
y_pred = model.predict(X_test_scaled)

# ------------------------------
# 7️⃣ Evaluate performance
# ------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)