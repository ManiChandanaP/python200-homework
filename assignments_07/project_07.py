

import os
import glob
import matplotlib
matplotlib.use("Agg")  

import pandas as pd
from dotenv import load_dotenv
from scipy import stats
from smolagents import CodeAgent, OpenAIServerModel, tool

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Shared global DataFrame — updated by load_happiness_data, read by all other tools.
# NOTE: smolagents' CodeAgent runs user-written code in a sandboxed interpreter that
# cannot see this module-level `df`. All tools therefore load from disk themselves
# when `df` is None, making each tool self-healing without requiring the agent's
# sandbox to reference the global directly.
df = None

# Resolve paths relative to this file so the script works regardless of
# which directory it is launched from.
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(
    os.path.join(_BASE, "..", "assignments_01", "outputs", "merged_happiness.csv")
)
FALLBACK_DIR = os.path.normpath(
    os.path.join(_BASE, "..", "assignments_01", "happiness_project")
)


def _ensure_loaded():
    """Internal helper: return the global df, loading it from disk if needed."""
    global df
    if df is not None:
        return df
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df
    csv_files = glob.glob(os.path.join(FALLBACK_DIR, "*.csv"))
    if not csv_files:
        return None
    frames = [pd.read_csv(f) for f in sorted(csv_files)]
    df = pd.concat(frames, ignore_index=True)
    return df


# Task 1: Tool Definitions


@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Attempts to load a pre-merged CSV from the default data path. If that file
    does not exist, falls back to loading and concatenating all yearly CSV files
    from the happiness_project resources directory. The resulting DataFrame is
    stored in the global variable `df` for use by other tools.

    Returns:
        dict: A dictionary with keys:
            - "shape" (tuple): Number of rows and columns as (rows, cols).
            - "columns" (list): List of column names in the loaded dataset.
            - "error" (str): Present only if loading failed entirely.
    """
    global df
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
        else:
            csv_files = glob.glob(os.path.join(FALLBACK_DIR, "*.csv"))
            if not csv_files:
                return {"error": f"No CSV files found at {DATA_PATH} or {FALLBACK_DIR}"}
            frames = [pd.read_csv(f) for f in sorted(csv_files)]
            df = pd.concat(frames, ignore_index=True)
        return {"shape": df.shape, "columns": df.columns.tolist()}
    except Exception as e:
        return {"error": str(e)}


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Uses pandas describe() to compute count, mean, std, min, quartiles, and max
    for the specified numeric column. The dataset must be loaded first via
    load_happiness_data.

    Args:
        column: The name of the column to summarize.

    Returns:
        dict: Descriptive statistics for the column, or a dict with an "error"
            key if the data is not loaded or the column is not found.
    """
    data = _ensure_loaded()
    if data is None:
        return {"error": f"No CSV files found at {DATA_PATH} or {FALLBACK_DIR}"}
    if column not in data.columns:
        return {"error": f"Column '{column}' not found. Available: {data.columns.tolist()}"}
    return data[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Uses scipy.stats.pearsonr to measure the linear relationship between two
    columns in the loaded dataset. A p-value below 0.05 is generally considered
    statistically significant. The dataset must be loaded first via
    load_happiness_data.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        dict: A dictionary with keys:
            - "col1" (str): Name of the first column.
            - "col2" (str): Name of the second column.
            - "pearson_r" (float): Pearson correlation coefficient, rounded to 4 decimal places.
            - "p_value" (float): Two-tailed p-value, rounded to 4 decimal places.
            Or a dict with an "error" key if the input is invalid or data is not loaded.
    """
    data = _ensure_loaded()
    if data is None:
        return {"error": f"No CSV files found at {DATA_PATH} or {FALLBACK_DIR}"}
    for col in [col1, col2]:
        if col not in data.columns:
            return {"error": f"Column '{col}' not found. Available: {data.columns.tolist()}"}
    try:
        valid = data[[col1, col2]].dropna()
        r, p = stats.pearsonr(valid[col1], valid[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Filters the loaded dataset to the specified year, sorts by the given column
    in descending order, and returns the top N rows. The dataset must be loaded
    first via load_happiness_data.

    Args:
        column: The name of the column to rank countries by (e.g., "happiness_score").
        year: The year to filter the data to (e.g., 2020).
        n: The number of top countries to return. Defaults to 5.

    Returns:
        dict: A dictionary with key "results" mapping to a list of dicts, each
            containing "country" and the value of the requested column. Or a
            dict with an "error" key if the input is invalid or data is not loaded.
    """
    data = _ensure_loaded()
    if data is None:
        return {"error": f"No CSV files found at {DATA_PATH} or {FALLBACK_DIR}"}
    if column not in data.columns:
        return {"error": f"Column '{column}' not found. Available: {data.columns.tolist()}"}
    if "year" not in data.columns:
        return {"error": "'year' column not found in dataset."}
    try:
        filtered = data[data["year"] == year].sort_values(column, ascending=False).head(n)
        if filtered.empty:
            return {"error": f"No data found for year {year}."}
        results = [
            {"country": row.get("country", row.get("Country", "Unknown")), column: row[column]}
            for _, row in filtered.iterrows()
        ]
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_happiness_data(columns: str, year: int = 0) -> dict:
    """Return happiness dataset rows as a list of records, optionally filtered by year.

    Use this tool when you need raw data for custom plotting or computation that
    the other tools do not cover — for example, drawing a line chart per region.
    The dataset must be loaded first via load_happiness_data.

    Args:
        columns: Comma-separated column names to include, e.g. "year,region,happiness_score".
                 Pass "all" to return every column.
        year: If non-zero, return only rows for that year. Defaults to 0 (all years).

    Returns:
        dict: A dictionary with key "records" mapping to a list of row dicts,
            or a dict with an "error" key if the data is not loaded or a column is missing.
    """
    data = _ensure_loaded()
    if data is None:
        return {"error": f"No CSV files found at {DATA_PATH} or {FALLBACK_DIR}"}
    try:
        if columns.strip().lower() == "all":
            subset = data.copy()
        else:
            col_list = [c.strip() for c in columns.split(",")]
            missing = [c for c in col_list if c not in data.columns]
            if missing:
                return {"error": f"Columns not found: {missing}. Available: {data.columns.tolist()}"}
            subset = data[col_list]
        if year:
            subset = subset[data["year"] == year]
        return {"records": subset.dropna().to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}



# Task 2: Build the Agent

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")


SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.

CRITICAL RULES you must follow every single step:
1. The variable `df` is NEVER available in your sandbox. Never reference it directly.
   To get raw data for plotting, always call get_happiness_data(columns="year,region,happiness_score")
   and build a local DataFrame with pd.DataFrame(result["records"]).
2. As soon as your task is complete, call final_answer() immediately with a short
   string or dict summarising what was done. Do NOT print anything after the work
   is finished — call final_answer() instead. Never skip final_answer().
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation,
           get_top_n_countries, get_happiness_data],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=[
        "pandas", "matplotlib", "matplotlib.pyplot", "scipy", "scipy.stats"
    ],
    max_steps=8,
)


# Task 3: Guided Queries


if __name__ == "__main__":
    os.makedirs(os.path.join(_BASE, "outputs"), exist_ok=True)

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        (
            "Plot happiness_score over the years as a line chart, with one line per region. "
            "Call get_happiness_data(columns='year,region,happiness_score') then build a "
            "DataFrame from result['records']. Use pivot_table(index='year', columns='region', "
            "values='happiness_score', aggfunc='mean') to get one mean score per region per year, "
            "then plot each column as a line. Save to outputs/happiness_by_region.png."
        ),
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    # Task 4: Custom Queries

    # My query 1 — asks the agent to compute correlations between multiple columns;
    # this is likely to trigger tool calls (compute_correlation) for each pair.
    my_query_1 = (
        "What are the correlations between social_support and happiness_score, "
        "and between freedom_to_make_life_choices and happiness_score? "
        "Which factor is more strongly associated with happiness?"
    )
    response_1 = agent.run(my_query_1, reset=False)
    print(f"\n--- My Query 1: {my_query_1} ---")
    print(response_1)
    # Comment: Did this trigger tool use, code generation, or both?
    # This triggered tool calls — the agent called compute_correlation twice (once per
    # column pair) and then synthesized the results into a comparison in text.

    # My query 2 — asks for a bar chart of the top 10 countries by happiness_score in 2019;
    # get_top_n_countries can supply the data, but plotting requires code generation.
    my_query_2 = (
        "Create a horizontal bar chart of the top 10 happiest countries in 2019 "
        "sorted from highest to lowest score, with country names on the y-axis. "
        "Save the chart to outputs/top10_2019.png."
    )
    response_2 = agent.run(my_query_2, reset=False)
    print(f"\n--- My Query 2: {my_query_2} ---")
    print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?
    # This triggered both tool use and code generation — the agent called
    # get_top_n_countries to fetch the data, then wrote matplotlib code to plot and save it.

    # --- Reflection ---
    #
    # 1. In Query 3, how did the agent communicate whether the correlation was statistically
    #    significant? Did it use the p-value correctly? What threshold did it apply?
    #
    #    The agent reported both the Pearson r value and the p-value returned by
    #    compute_correlation, then stated whether the result was "statistically significant."
    #    It correctly applied the conventional 0.05 threshold: since the p-value for
    #    gdp_per_capita vs happiness_score is extremely small (effectively 0.0), the agent
    #    concluded the correlation is highly significant. The use of the p-value was accurate
    #    — it did not just report the magnitude of r but explicitly tied significance to the
    #    p-value being below 0.05.
    #
    # 2. Did any of the agent's responses surprise you — either by being more capable than
    #    you expected, or less? Describe one specific example.
    #
    #    Query 5 (the regional line chart) was more impressive than expected. I assumed the
    #    agent might struggle because no tool covers multi-line regional plots — it would need
    #    to figure out the right groupby logic, choose a color palette, and handle the
    #    matplotlib API all on its own. Instead it wrote clean, correct code: it called
    #    get_happiness_data, built a local DataFrame, used pivot_table to reshape by region,
    #    and plotted one line per region with a legend. It saved the file exactly where
    #    specified. That level of autonomous code generation felt genuinely capable.
    #
    # 3. What one additional tool would make this agent meaningfully more useful?
    #    Describe what it would do and what kind of question it would help the agent answer.
    #
    #    A filter_by_region(region: str) tool would be very valuable. It would return a
    #    subset of df containing only rows matching the given region name, allowing the agent
    #    to answer questions like "Which country improved happiness the most in Western Europe
    #    between 2015 and 2020?" or "What is the average GDP per capita in Sub-Saharan Africa?"
    #    without writing custom filtering code every time. Right now the agent writes a pandas
    #    mask inline whenever a region-specific question comes up, which works but is fragile
    #    if region names don't match exactly. A dedicated tool could also normalize region
    #    name casing and handle partial matches, making regional analysis much more reliable.