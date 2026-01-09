import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("dataset/winequality-red.csv", sep=';')

selected_features = [
    "alcohol",
    "volatile acidity",
    "sulphates",
    "citric acid",
    "density"
]

X = data[selected_features]
y = data["quality"]

X_scaled = X.values

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")

joblib.dump(model, "output/model.pkl")

results = {
    "mse": mse,
    "r2_score": r2
}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)
