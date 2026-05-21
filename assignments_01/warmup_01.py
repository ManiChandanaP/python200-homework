#Question1 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)
print(df)
print(f"Num Rows: {len(df)}")
print(df.shape)


#Question2
print(df[(df["passed"]) & (df["grade"] > 80)])

#Question3
df['grade_curved'] = df['grade']+5
print(df)

#Question4
df['name_upper'] = df['name'].str.upper()
print(df)

#Question5
print(df.groupby('city')['grade'].mean())

#Question6
df['city'] = df['city'].replace('Austin','Houston')
print(df[['name','city']])

#Question7
df_sorted = df.sort_values(by="grade", ascending=False)
print(df_sorted.head(3))

#Numpy
#Q1
arr = np.array([10, 20, 30, 40, 50])
print(arr.dtype)
print(arr.shape)
print(arr.ndim)

#Q2
arr1 = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(arr1.size)
print(arr1.shape)

#Q3
top_left = arr1[:2,:2]
print(top_left)

#Q4
zero_array = np.zeros((3,4))
ones_array = np.ones((2,5))
print(f"Zero Array\n",zero_array)
print(f"Ones Array\n",ones_array)

#Q5

ar = np.arange(0, 50, 5)
print(ar)
print(ar.sum())
print(ar.mean())
print(ar.shape)
print(ar.std())

#Q6
random_ar = np.random.normal(0,1,200)
print(random_ar)
print(random_ar.mean())
print(random_ar.std())


#Matplotlib
#Q1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.figure()
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#Q2
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

plt.figure()
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

#Q3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.figure()
plt.scatter(x1, y1, color="blue", label="Dataset 1")
plt.scatter(x2, y2, color="red", label="Dataset 2")
plt.title("Scatter Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#Q4
fig, ax = plt.subplots(1, 2)
# Left subplot (line plot)
ax[0].plot(x, y)
ax[0].set_title("Squares")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")

# Right subplot (bar plot)
ax[1].bar(subjects, scores)
ax[1].set_title("Subject Scores")
ax[1].set_xlabel("Subjects")
ax[1].set_ylabel("Scores")

plt.tight_layout()
plt.show()


#Descriptive
#Q1
data = [[12, 15, 14, 10, 18, 22, 13, 16, 14, 15]]
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Variance: {np.var(data)}")
print(f"Standard Deviation: {np.std(data)}")

#Q2
values = np.random.normal(65, 10, 500)
plt.figure()
plt.hist(values, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

#Q3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.figure()
plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.show()

#Q4
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.figure()
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.show()
# Comment:
# The exponential distribution is more skewed (right-skewed).
# For skewed data, the median is a better measure of central tendency.
# For normally distributed data, the mean is appropriate.

#Q5
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print("\nData1:")
print(f"Mean: {np.mean(data1)}")
print(f"Median: {np.median(data1)}")
print(f"Mode: {stats.mode(data1)[0]}")

print("\nData2:")
print(f"Mean: {np.mean(data2)}")
print(f"Median: {np.median(data2)}")
print(f"Mode: {stats.mode(data2)[0]}")
# Comment:
# The mean for data2 is much higher because of the extreme outlier (150).
# The median is less affected by outliers, so it stays closer to the center
# of the majority of the data.

#Hypothesis
#Question 1
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
t_stat, p_val = stats.ttest_ind(group_a, group_b)
print(f"Independent t-test:\nt-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")


#Question 2
alpha = 0.05
if p_val < alpha:
    print("Result is statistically significant at alpha = 0.05")
else:
    print("Result is NOT statistically significant at alpha = 0.05")


#Question 3
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]
t_stat_paired, p_val_paired = stats.ttest_rel(before, after)
print(f"\nPaired t-test:\nt-statistic = {t_stat_paired:.4f}, p-value = {p_val_paired:.4f}")


#Question 4
scores = [72, 68, 75, 70, 69, 74, 71, 73]
benchmark = 70
t_stat_one, p_val_one = stats.ttest_1samp(scores, benchmark)
print(f"\nOne-sample t-test:\nt-statistic = {t_stat_one:.4f}, p-value = {p_val_one:.4f}")


#Question 5
t_stat_one_tailed, p_val_one_tailed = stats.ttest_ind(group_a, group_b, alternative='less')
print(f"\nOne-tailed t-test (group_a < group_b) p-value = {p_val_one_tailed:.4f}")

# Question 6: 
print("\nConclusion for independent t-test (Q1):")
print("Group A has lower scores than Group B, and the difference is statistically significant. "
      "It is unlikely that this difference is due to random chance.")

#Correlation
#Question 1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)
print("Full correlation matrix:")
print(corr_matrix)

print("Correlation coefficient (x vs y):", corr_matrix[0, 1])


#Question 2
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

r, p_value = pearsonr(x, y)
print(f"\nPearson correlation coefficient: {r}")
print(f"P-value: {p_value}")


#Question 3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}

df = pd.DataFrame(people)
corr_df = df.corr()
print("\nCorrelation matrix:")
print(corr_df)


#Question 4
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.figure()
plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


#Question 5
plt.figure()
sns.heatmap(corr_df, annot=True)
plt.title("Correlation Heatmap")
plt.show()


#Pipelines
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan,
                18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# Step 1: Create Series
def create_series(arr):
    return pd.Series(arr, name="values")

# Step 2: Clean data
def clean_data(series):
    return series.dropna()

# Step 3: Summarize data
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

# Pipeline function
def data_pipeline(arr):
    series = create_series(arr)
    clean_series = clean_data(series)
    summary = summarize_data(clean_series)
    return summary

# Run pipeline
result = data_pipeline(arr)

# Print results
for key, value in result.items():
    print(f"{key}: {value}")