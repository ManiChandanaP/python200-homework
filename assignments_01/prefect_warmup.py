import numpy as np
import pandas as pd
from prefect import task, flow

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan,
                18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# Step 1: Create Series
@task
def create_series(arr):
    return pd.Series(arr, name="values")

# Step 2: Clean data
@task
def clean_data(series):
    return series.dropna()

# Step 3: Summarize data
@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

# Flow
@flow
def pipeline_flow():
    series = create_series(arr)
    clean_series = clean_data(series)
    summary = summarize_data(clean_series)

    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary


# Run flow
if __name__ == "__main__":
    pipeline_flow()


# --- Reflection Questions ---
# Why Prefect might be overkill here:
# This pipeline is very simple, with only a few small functions and a tiny dataset.
# There is no need for scheduling, retries, monitoring, or orchestration.
# Using Prefect adds extra complexity without providing much benefit.

# When Prefect would be useful:
# Prefect is valuable in real-world scenarios such as:
# - Large datasets that take time to process
# - Pipelines that depend on external APIs or databases (which may fail)
# - Scheduled workflows (daily/weekly jobs)
# - Multi-step pipelines with dependencies between tasks
# - Situations where logging, monitoring, and retries are important