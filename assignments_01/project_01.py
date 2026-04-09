import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from prefect import task, flow, get_run_logger
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "assignments_01", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Task 1: Load Data
@task(retries=3, retry_delay_seconds=2)
def load_data():
    logger = get_run_logger()
    dfs = []
    for year in range(2015, 2025):
        file_path = os.path.join(
            BASE_DIR,
            "happiness_project",
            f"world_happiness_{year}.csv"
        )
        logger.info(f"Loading: {file_path}")
        df = pd.read_csv(file_path, sep=";", decimal=",")

        df.columns = df.columns.str.lower().str.replace(" ", "_")
        if "regional_indicator" in df.columns:
            df["region"] = df["regional_indicator"]

        if "country_name" in df.columns:
            df["country"] = df["country_name"]

        if "country" not in df.columns:
            df.rename(columns={"country_or_region": "country"}, inplace=True)

        if "life_ladder" in df.columns:
            df["happiness_score"] = df["life_ladder"]

        df["year"] = year
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    combined.to_csv(os.path.join(OUTPUT_DIR, "merged_happiness.csv"), index=False)
    logger.info("Merged dataset saved successfully.")

    return combined

# Task 2: Descriptive Stats
@task
def descriptive_stats(df):
    logger = get_run_logger()

    logger.info(f"Mean happiness: {df['happiness_score'].mean()}")
    logger.info(f"Median happiness: {df['happiness_score'].median()}")
    logger.info(f"Std happiness: {df['happiness_score'].std()}")

    by_year = df.groupby("year")["happiness_score"].mean()
    by_region = df.groupby("region")["happiness_score"].mean()

    logger.info(f"Mean by year:\n{by_year}")
    logger.info(f"Mean by region:\n{by_region}")

    return by_region

# Task 3: Visualizations
@task
def create_plots(df):
    logger = get_run_logger()

    # Histogram
    plt.figure()
    plt.hist(df["happiness_score"], bins=20)
    plt.title("Happiness Score Distribution")
    plt.savefig(os.path.join(OUTPUT_DIR, "happiness_histogram.png"))
    plt.close()
    logger.info("Saved histogram")

    # Boxplot
    plt.figure()
    df.boxplot(column="happiness_score", by="year")
    plt.title("Happiness by Year")
    plt.suptitle("")
    plt.savefig(os.path.join(OUTPUT_DIR, "happiness_by_year.png"))
    plt.close()
    logger.info("Saved boxplot")

    # Scatter
    if "gdp_per_capita" in df.columns:
        plt.figure()
        plt.scatter(df["gdp_per_capita"], df["happiness_score"])
        plt.xlabel("GDP per Capita")
        plt.ylabel("Happiness Score")
        plt.savefig(os.path.join(OUTPUT_DIR, "gdp_vs_happiness.png"))
        plt.close()
        logger.info("Saved scatter plot")

    # Heatmap
    plt.figure()
    numeric_df = df.select_dtypes(include=np.number)
    sns.heatmap(numeric_df.corr(), annot=True)
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plt.close()
    logger.info("Saved heatmap")


# Task 4: Hypothesis Testing
@task
def hypothesis_tests(df):
    logger = get_run_logger()

    data_2019 = df[df["year"] == 2019]["happiness_score"].dropna()
    data_2020 = df[df["year"] == 2020]["happiness_score"].dropna()

    t_stat, p_val = stats.ttest_ind(data_2019, data_2020)

    logger.info(f"T-statistic: {t_stat}")
    logger.info(f"P-value: {p_val}")
    logger.info(f"Mean 2019: {data_2019.mean()}")
    logger.info(f"Mean 2020: {data_2020.mean()}")

    if p_val < 0.05:
        logger.info(
            "There is a statistically significant difference in happiness between 2019 and 2020, suggesting the pandemic may have impacted global happiness."
        )
    else:
        logger.info(
            "There is no statistically significant difference in happiness between 2019 and 2020, suggesting the pandemic did not produce a measurable global shift."
        )

    # Second test: top vs bottom region
    regions = df.groupby("region")["happiness_score"].mean().sort_values()
    low_region = regions.index[0]
    high_region = regions.index[-1]

    r1 = df[df["region"] == low_region]["happiness_score"]
    r2 = df[df["region"] == high_region]["happiness_score"]

    t2, p2 = stats.ttest_ind(r1, r2)

    logger.info(f"Comparing {low_region} vs {high_region}")
    logger.info(f"T-stat: {t2}, P-value: {p2}")


# Task 5: Correlation 
@task
def correlation_analysis(df):
    logger = get_run_logger()
    numeric_df = df.select_dtypes(include=np.number)
    results = []
    for col in numeric_df.columns:
        if col != "happiness_score":
            try:
                clean_df = df[[col, "happiness_score"]].dropna()
                r, p = stats.pearsonr(clean_df[col], clean_df["happiness_score"])
                results.append((col, r, p))
            except:
                continue

    alpha = 0.05
    adjusted_alpha = alpha / len(results)

    logger.info(f"Adjusted alpha: {adjusted_alpha}")

    for col, r, p in results:
        logger.info(
            f"{col}: r={r:.3f}, p={p:.5f}, "
            f"significant={p < alpha}, after_correction={p < adjusted_alpha}"
        )

    strongest = max(results, key=lambda x: abs(x[1]))
    return strongest, results, adjusted_alpha

# Task 6: Summary
@task
def summary(df, by_region, corr_output):
    logger = get_run_logger()

    strongest_corr, results, adjusted_alpha = corr_output

    logger.info(f"Total countries: {df['country'].nunique()}")
    logger.info(f"Total years: {df['year'].nunique()}")

    top3 = by_region.sort_values(ascending=False).head(3)
    bottom3 = by_region.sort_values().head(3)

    logger.info(f"Top 3 regions by happiness:\n{top3}")
    logger.info(f"Bottom 3 regions by happiness:\n{bottom3}")

    logger.info(
        f"Strongest correlation: {strongest_corr[0]} "
        f"(r={strongest_corr[1]:.3f}, p={strongest_corr[2]:.5f})"
    )


@flow
def happiness_pipeline():
    df = load_data()
    by_region = descriptive_stats(df)
    create_plots(df)
    hypothesis_tests(df)
    corr_output = correlation_analysis(df)
    summary(df, by_region, corr_output)
    
if __name__ == "__main__":
    happiness_pipeline()