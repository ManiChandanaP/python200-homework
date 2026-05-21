from dotenv import load_dotenv
load_dotenv()

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from scipy.stats import pearsonr

from smolagents import tool, CodeAgent, OpenAIServerModel



df = None

DATA_PATH = "assignments_01/outputs/merged_happiness.csv"
FALLBACK_DIR = "assignments_01/happiness_project/"
OUTPUT_DIR = "assignments_07/outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@tool
def load_happiness_data() -> dict:
    """
    Loads dataset and RETURNS it (not global state).
    """

    if os.path.exists(DATA_PATH):
        data = pd.read_csv(DATA_PATH)
    else:
        files = glob.glob(os.path.join(FALLBACK_DIR, "*.csv"))

        dfs = []
        for f in files:
            temp = pd.read_csv(f)
            temp["year"] = os.path.basename(f).replace(".csv", "")
            dfs.append(temp)

        data = pd.concat(dfs, ignore_index=True)

    data.columns = [c.strip() for c in data.columns]

    return {
        "shape": list(data.shape),
        "columns": list(data.columns),
        "data": data.to_dict(orient="records")   
    }

@tool
def summarize_column(data: list, column: str) -> dict:
    """
    Summarize a column from a dataset.

    Args:
        data (list): The dataset as a list of dictionaries (rows).
        column (str): The column name to analyze.

    Returns:
        dict: Summary statistics for the column or an error message.
    """
    df = pd.DataFrame(data)

    if column not in df.columns:
        return {"error": "Column not found"}

    return df[column].describe().to_dict()

@tool
def compute_correlation(data: list, col1: str, col2: str) -> dict:
    """
    Compute Pearson correlation between two columns.

    Args:
        data (list): Dataset as list of dictionaries.
        col1 (str): First column.
        col2 (str): Second column.

    Returns:
        dict: correlation result or error.
    """
    import pandas as pd
    from scipy.stats import pearsonr

    df = pd.DataFrame(data)

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "Column not found"}

    temp = df[[col1, col2]].dropna()

    if len(temp) < 2:
        return {"error": "Not enough data"}

    r, p = pearsonr(temp[col1], temp[col2])

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4)
    }


@tool
def get_top_n_countries(data: list, column: str, year: int, n: int = 5) -> dict:
    """
    Get top N countries.

    Args:
        data (list): dataset
        column (str): metric column
        year (int): year filter
        n (int): number of results

    Returns:
        dict
    """
    import pandas as pd

    df = pd.DataFrame(data)

    if "year" not in df.columns or "country" not in df.columns:
        return {"error": "Missing required columns"}

    temp = df[df["year"].astype(str) == str(year)]

    if temp.empty:
        return {"error": "No data for year"}

    temp = temp[["country", column]].dropna()
    temp = temp.sort_values(by=column, ascending=False).head(n)

    return {
        "results": temp.to_dict(orient="records")
    }


api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini"
)

SYSTEM_PROMPT = """
CRITICAL RULES:

1. load_happiness_data() RETURNS a dictionary with a "data" key.
2. To use the dataset, ALWAYS do:

df = pd.DataFrame(result["data"])

3. NEVER use global df.
4. NEVER assume df exists.
5. Every query must first load data via tool.

Be strict and consistent.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


if __name__ == "__main__":

    queries = [
        "Load the data and show shape and columns",
        "Summarize happiness_score",
        "Correlation between gdp_per_capita and happiness_score",
        "Top 5 happiest countries in 2020",
        "Plot happiness_score over years by region and save it to outputs/happiness_by_region.png"
    ]

    for q in queries:
        print("\n---", q, "---")
        print(agent.run(q, reset=False))


    # Custom queries
    print("\nQ1:", agent.run("Which country has highest happiness_score overall?", reset=False))

    print("\nQ2:", agent.run(
        "Create scatter plot of gdp_per_capita vs happiness_score colored by region and save it",
        reset=False
    ))


"""
1. The agent uses p-value from pearsonr and typically checks against 0.05.

2. It can generate correct matplotlib grouping plots without tools.

3. Useful extra tool: automated time-series plotting by group.
"""