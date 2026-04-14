import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("assignments_02/outputs", exist_ok=True)

# Task 1: Load and Explore
df = pd.read_csv("assignments_02/student_performance_math.csv", sep=";")

print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nData types:\n", df.dtypes)

# Histogram of G3
plt.figure()
plt.hist(df["G3"], bins=21)
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Count")
plt.savefig("assignments_02/outputs/g3_distribution.png")
plt.close()


# Task 2: Preprocess
# Remove G3 = 0 (students who missed final exam)
df_clean = df[df["G3"] > 0].copy()

print("\nOriginal shape:", df.shape)
print("Filtered shape:", df_clean.shape)

# Convert yes/no to 1/0
binary_cols = ["schoolsup", "internet", "higher", "activities"]
for col in binary_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

# Convert sex to 0/1
df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

# Correlation absences vs G3
corr_original = df["absences"].corr(df["G3"])
corr_filtered = df_clean["absences"].corr(df_clean["G3"])

print("\nCorrelation (original):", corr_original)
print("Correlation (filtered):", corr_filtered)


# Task 3: EDA

numeric_cols = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "absences", "freetime", "goout", "Walc"
]

correlations = df_clean[numeric_cols + ["G3"]].corr()["G3"].drop("G3").sort_values()

print("\nCorrelations with G3:\n", correlations)

# Plot 1: failures vs G3
plt.figure()
plt.scatter(df_clean["failures"], df_clean["G3"])
plt.xlabel("Failures")
plt.ylabel("G3")
plt.title("Failures vs Final Grade")
plt.savefig("assignments_02/outputs/failures_vs_g3.png")
plt.close()

# Plot 2: studytime vs G3
plt.figure()
plt.scatter(df_clean["studytime"], df_clean["G3"])
plt.xlabel("Study Time")
plt.ylabel("G3")
plt.title("Study Time vs Final Grade")
plt.savefig("assignments_02/outputs/studytime_vs_g3.png")
plt.close()

# Task 4: Baseline Model
X = df_clean[["failures"]].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nBaseline Model:")
print("Slope:", model.coef_[0])
print("RMSE:", rmse)
print("R2:", r2)

# Task 5: Full Model
feature_cols = [
    "failures", "Medu", "Fedu", "studytime", "higher",
    "schoolsup", "internet", "sex", "freetime",
    "activities", "traveltime"
]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nFull Model:")
print("Train R2:", train_r2)
print("Test R2:", test_r2)
print("RMSE:", rmse)

print("\nCoefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")


# Task 6: Predicted vs Actual
plt.figure()
plt.scatter(y_pred, y_test)
plt.plot([0, 20], [0, 20], color="red")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Predicted vs Actual (Full Model)")
plt.savefig("assignments_02/outputs/predicted_vs_actual.png")
plt.close()


# Neglected Feature: Add G1
feature_cols_with_g1 = feature_cols + ["G1"]

X = df_clean[feature_cols_with_g1].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

test_r2_g1 = model.score(X_test, y_test)

print("\nModel with G1 included:")
print("Test R2:", test_r2_g1)