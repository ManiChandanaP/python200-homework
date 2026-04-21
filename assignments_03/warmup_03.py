import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

#Q1
X_train, X_test,y_train, y_test = train_test_split(
    X,y, test_size=0.2,stratify=y,random_state=42
)
print("Q1: Shapes of datasets")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#Q2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n Scaled data:")
print(np.mean(X_train_scaled, axis=0))

# Explanation:
# We fit the scaler only on X_train to prevent data leakage,
# ensuring that information from the test set does not influence the model.

#KNN
#Q1
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("\nKNN Q1: Unscaled Data")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Q2
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)
print("\nKNN Q2: Scaled Data")
print("Accuracy:", accuracy_score(y_test, y_pred_scaled))

# Comment:
# Scaling may slightly improve performance because KNN relies on distance calculations, 
# and features in the Iris dataset have different scales. Standardizing ensures all
# features contribute equally, though the effect may be small since the feature ranges
# are already fairly similar.


# Q3
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

print("\nKNN Q3: 5-Fold Cross-Validation (Unscaled)")
print("Fold scores:", cv_scores)
print("Mean CV score:", np.mean(cv_scores))
print("Standard deviation:", np.std(cv_scores))

# Comment:
# This result is more trustworthy than a single train/test split because it averages
# performance across multiple splits, reducing the impact of any one lucky or unlucky split.


# Q4
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

print("\nKNN Q4: k vs CV Accuracy (Unscaled)")
for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_k, X_train, y_train, cv=5)
    print(f"k={k}, Mean CV Accuracy={np.mean(scores):.4f}")

# Comment:
# Choose the k with the highest mean CV accuracy. Typically, a moderate k (like 5–9)
# balances bias and variance: small k can overfit, while large k can underfit.

#Classifier Evaluation
# Q1

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

disp.plot()
plt.title("KNN Confusion Matrix (Unscaled)")
plt.savefig("outputs/knn_confusion_matrix.png")
plt.close()

# Comment:
# The model most often confuses versicolor and virginica, which are known to be
# harder to separate because their feature distributions overlap.


#Decision Trees
# Q1

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print("\nDecision Tree Q1")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Classification Report:\n", classification_report(y_test, y_pred_tree))

# Comment:
# The Decision Tree accuracy is typically similar to or slightly lower than KNN,
# depending on the split. KNN can capture local structure better, while a shallow
# tree may underfit.

# Comment:
# Scaling does not meaningfully affect Decision Trees because they split on feature
# thresholds rather than distance calculations.


#Logistic Regression
# Q1

C_values = [0.01, 1.0, 100]
print("\nLogistic Regression Q1: Coefficient Magnitudes")
for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000, solver='liblinear')
    model.fit(X_train_scaled, y_train)
    coef_magnitude = np.abs(model.coef_).sum()
    print(f"C={C}, Total |coefficients|={coef_magnitude:.4f}")

# Comment:
# As C increases, the total coefficient magnitude increases. This shows that
# weaker regularization (larger C) allows the model to use larger weights,
# while stronger regularization (smaller C) shrinks coefficients toward zero.


#PCA 
digits = load_digits()
X_digits = digits.data
y_digits = digits.target
images = digits.images
# Q1
print("\nPCA Q1:")
print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)

fig, axes = plt.subplots(1, 10, figsize=(12, 2))

for digit in range(10):
    idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[idx], cmap='gray_r')
    axes[digit].set_title(str(digit))
    axes[digit].axis('off')

plt.tight_layout()
plt.savefig("outputs/sample_digits.png")
plt.close()


# Q2
pca = PCA()
pca.fit(X_digits)

scores = pca.transform(X_digits)

plt.figure(figsize=(6, 5))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)
plt.colorbar(scatter, label='Digit')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D Projection")
plt.savefig("outputs/pca_2d_projection.png")
plt.close()

# Comment:
# Yes, many digits form clusters in this 2D space, though some overlap exists


# Q3

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure()
plt.plot(cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Variance Explained")
plt.savefig("outputs/pca_variance_explained.png")
plt.close()

# Comment:
# Roughly 20–30 components are needed to explain about 80% of the variance.


# Q4

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)


n_values = [2, 5, 15, 40]

fig, axes = plt.subplots(len(n_values) + 1, 5, figsize=(10, 8))

for i in range(5):
    axes[0, i].imshow(images[i], cmap='gray_r')
    axes[0, i].set_title(f"Orig {y_digits[i]}")
    axes[0, i].axis('off')


for row, n in enumerate(n_values, start=1):
    for col in range(5):
        recon = reconstruct_digit(col, scores, pca, n)
        axes[row, col].imshow(recon, cmap='gray_r')
        axes[row, col].set_title(f"n={n}")
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png")
plt.close()

# Comment:
# Digits become clearly recognizable around n ≈ 10–20 components, which aligns
# with where the cumulative variance curve begins to level off.

