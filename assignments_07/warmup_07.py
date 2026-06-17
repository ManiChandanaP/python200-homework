from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

from smolagents import (
    ToolCallingAgent,
    CodeAgent,
    OpenAIServerModel,
    tool
)

client = OpenAI()


CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bike_commute.csv")

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


# JSON schema
celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "Temperature in Celsius"
                }
            },
            "required": ["celsius"]
        }
    }
}

print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


# Q2
"""
Prediction:
- The query "Convert 100 degrees Celsius to Fahrenheit"
  will NOT trigger a tool call because the only available
  tool is get_current_time.
- The model already knows how to convert temperatures.
- Only one API call should happen.
"""


def get_current_time():
    """Return the current local time."""
    return datetime.now().strftime("%H:%M:%S")


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local time.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


def run_agent(user_query):
    messages = [{"role": "user", "content": user_query}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )

    message = response.choices[0].message

    if message.tool_calls:

        tool_call = message.tool_calls[0]

        if tool_call.function.name == "get_current_time":

            tool_result = get_current_time()

            messages.append(message)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

            second_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

            return second_response.choices[0].message.content

    return message.content


result_q2 = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print(result_q2)

# Prediction was correct:
# No tool was called because the available tool
# had nothing to do with temperature conversion.

# Q3

tools_extended = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local time.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    celsius_to_fahrenheit_schema
]


def run_agent_extended(user_query):

    messages = [{"role": "user", "content": user_query}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools_extended
    )

    message = response.choices[0].message

    if message.tool_calls:

        tool_call = message.tool_calls[0]

        tool_name = tool_call.function.name

        args = json.loads(tool_call.function.arguments)

        if tool_name == "get_current_time":
            tool_result = get_current_time()

        elif tool_name == "celsius_to_fahrenheit":
            tool_result = celsius_to_fahrenheit(args["celsius"])

        else:
            tool_result = "Unknown tool"

        messages.append(message)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(tool_result)
        })

        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        return second_response.choices[0].message.content

    return message.content


response_a = run_agent_extended(
    "What is 37 degrees Celsius in Fahrenheit?"
)

print("Response A:", response_a)

# Tool WAS called because the query directly matched
# the celsius_to_fahrenheit tool.


response_b = run_agent_extended(
    "What is the boiling point of water in plain English?"
)

print("Response B:", response_b)

# No tool was needed because the model could answer
# conversationally using general knowledge.


class CsvManager:

    def __init__(self):
        self.df = None

    def load_csv(self, filepath: str):

        try:
            self.df = pd.read_csv(filepath)

            return {
                "status": "success",
                "columns": list(self.df.columns),
                "rows": len(self.df)
            }

        except Exception as e:
            return {"error": str(e)}

    def preview_data(self, rows: int = 5):

        if self.df is None:
            return {"error": "No CSV loaded."}

        return self.df.head(rows).to_dict()

    def summarize_data(self):

        if self.df is None:
            return {"error": "No CSV loaded."}

        return self.df.describe(include="all").to_dict()

    def plot_data(self, x_col: str, y_col: str):

        if self.df is None:
            return {"error": "No CSV loaded."}

        if x_col not in self.df.columns or y_col not in self.df.columns:
            return {"error": "Column not found."}

        plt.figure(figsize=(8, 5))

        plt.scatter(
            self.df[x_col],
            self.df[y_col],
            color="green"
        )

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{y_col} vs {x_col}")

        plt.show()

        return {"status": "plot displayed"}

    def compute_correlation(self, col1: str, col2: str):

        """
        Compute Pearson correlation between two columns.
        """

        if self.df is None:
            return {"error": "No CSV loaded."}

        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {"error": "Column not found."}

        try:

            r, p = pearsonr(
                self.df[col1],
                self.df[col2]
            )

            return {
                "col1": col1,
                "col2": col2,
                "pearson_r": round(float(r), 4),
                "p_value": round(float(p), 4)
            }

        except Exception as e:
            return {"error": str(e)}

    def list_csv_files(self):

        """
        List all CSV files in current directory.
        """

        return [
            file for file in os.listdir()
            if file.endswith(".csv")
        ]


csv_manager = CsvManager()

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string"
                    }
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute Pearson correlation between two columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string"},
                    "col2": {"type": "string"}
                },
                "required": ["col1", "col2"]
            }
        }
    }
]

node_tools = {
    "load_csv": csv_manager.load_csv,
    "compute_correlation": csv_manager.compute_correlation
}

SYSTEM_PROMPT = """
You are a CSV analysis assistant.
Use tools whenever necessary.
"""


def run_agent_cycle(messages, user_input, max_rounds=5):

    messages.append({
        "role": "user",
        "content": user_input
    })

    for _ in range(max_rounds):

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_schema
        )

        message = response.choices[0].message
        messages.append(message.model_dump())

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            args = json.loads(
                tool_call.function.arguments
            )

            tool_function = node_tools[tool_name]

            result = tool_function(**args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

    return "Tool round limit reached."


messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]

result = run_agent_cycle(
    messages,
    f"Load {CSV_PATH} and compute the correlation between avg_traffic_density and avg_speed_kmh."
)

print(result)

"""
Role explanations:

- system:
  Gives the assistant instructions and behavior rules.

- user:
  The user's question or request.

- assistant:
  The model's reasoning and tool-call decisions.

- tool:
  The outputs returned from external functions/tools.
"""

print(json.dumps(messages, indent=2, default=str))


@tool
def compute_correlation_tool(
    col1: str,
    col2: str
) -> dict:
    """
    Compute Pearson correlation between two CSV columns.

    Args:
        col1: First column name.
        col2: Second column name.
    """

    return csv_manager.compute_correlation(
        col1,
        col2
    )


@tool
def load_csv_tool(
    filepath: str
) -> dict:
    """
    Load a CSV file.

    Args:
        filepath: Path to the CSV file.
    """

    return csv_manager.load_csv(filepath)


@tool
def plot_data_tool(
    x_col: str,
    y_col: str
) -> dict:
    """
    Plot two columns from the loaded CSV.

    Args:
        x_col: X-axis column.
        y_col: Y-axis column.
    """

    return csv_manager.plot_data(
        x_col,
        y_col
    )


print(compute_correlation_tool.description)

# smolagents automatically creates tool descriptions
# using the function name, type hints, and docstring.
#
# Unlike the manual JSON schema from Q4,
# we do not manually define parameter structures.
#
# smolagents depends on:
# - clear type hints
# - good docstrings
# - meaningful function names
#
# Better developer documentation produces
# better autogenerated schemas/descriptions.


TOOLS = [
    load_csv_tool,
    plot_data_tool,
    compute_correlation_tool
]

model = OpenAIServerModel(
    model_id="gpt-4o-mini"
)

tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model
)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model
)

prompt = f"""
Load {CSV_PATH}.
Plot avg_heart_rate vs duration_min as a scatter plot with green dots.
"""

response_tool = tool_agent.run(prompt)

print("\nToolCallingAgent Response:")
print(response_tool)

response_code = code_agent.run(
    prompt,
    additional_args={
        "csv_manager": csv_manager
    }
)

print("\nCodeAgent Response:")
print(response_code)

"""
Comparison:

- ToolCallingAgent:
  Uses only predefined tools.
  It is more limited and structured.

- CodeAgent:
  Generates and executes Python code.
  It can customize plots and logic more flexibly.

The ToolCallingAgent may not fully customize
the plot styling unless the tool itself supports it.

The CodeAgent can directly write matplotlib code
to create green scatter plot dots.

This shows:
- ToolCallingAgents are better for safe,
  predictable workflows.

- CodeAgents are better for flexible,
  open-ended programming tasks.
"""

"""
1. When is a ToolCallingAgent better?

A ToolCallingAgent is better for structured tasks
where the allowed actions are predefined,
such as:
- checking weather,
- querying databases,
- booking tickets,
- retrieving account information.

These tasks are safer because the agent is restricted
to approved tools only.


2. One risk of a CodeAgent:

A CodeAgent dynamically generates and executes code.

This creates risks such as:
- unsafe code execution,
- accidental file modification,
- security vulnerabilities,
- unpredictable behavior,
- excessive resource usage.

A ToolCallingAgent avoids these risks because
it can only call approved tools/functions.
"""