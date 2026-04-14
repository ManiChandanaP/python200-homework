import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

#Q1
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
model = LinearRegression()
model.fit(years,salary)
pred_4 = model.predict([[4]])[0]
pred_8 = model.predict([[8]])[0]
print("Slope (coef):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted salary for 4 years:", pred_4)
print("Predicted salary for 8 years:", pred_8)

#Q2
x = np.array([10, 20, 30, 40, 50])
print("Original data",x.shape)

x_2d=x.reshape(-1,1)
print("2D data",x_2d.shape)

# Explanation:
# scikit-learn expects X to be 2D because it treats data as a table:
# rows = samples (data points), columns = features.
# Even if there is only one feature, it still needs a column structure.

#Q3
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)
print("Cluster Centers:\n", kmeans.cluster_centers_)
counts = np.bincount(labels)
print("Points per cluster:", counts)
os.makedirs("assignments_02/outputs", exist_ok=True)
plt.figure()
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', s=50)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=200)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("assignments_02/outputs/kmeans_clusters.png")
plt.close()



#Linear Regression
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

os.makedirs("assignments_02/outputs", exist_ok=True)

# Question 1: Scatter Plot
plt.figure()
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.savefig("assignments_02/outputs/cost_vs_age.png")
plt.close()

# Comment:
# We see two distinct groups: smokers (higher cost) and non-smokers (lower cost).
# This suggests that smoking has a strong impact on medical cost.


# Question 2
X = age.reshape(-1, 1)
y = cost

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Question 3: Linear Regression (age only)
model = LinearRegression()
model.fit(X_train, y_train)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print("RMSE:", rmse)
print("R²:", r2)

# Comment:
# The slope represents how much cost increases per additional year of age.
# For example, a slope near 200 means each extra year adds about $200 in medical costs.

# Question 4: Add smoker feature
X_full = np.column_stack([age, smoker])

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)

r2_full = model_full.score(X_test_f, y_test_f)

print("R² (age only):", r2)
print("R² (age + smoker):", r2_full)

print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# Comment:
# Adding smoker improves R² significantly, meaning the model explains the data better.
# The smoker coefficient (~15000) means smokers pay about $15,000 more per year
# than non-smokers, on average, holding age constant.


# Question 5: Predicted vs Actual Plot
y_pred_full = model_full.predict(X_test_f)

plt.figure()
plt.scatter(y_pred_full, y_test_f)

# Diagonal line
min_val = min(y_test_f.min(), y_pred_full.min())
max_val = max(y_test_f.max(), y_pred_full.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--')

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")

plt.savefig("assignments_02/outputs/predicted_vs_actual.png")
plt.close()

# Comment:
# Points above the diagonal → actual cost > predicted (model underestimates).
# Points below the diagonal → actual cost < predicted (model overestimates).