
# Classifying weather conditions for running is not an ideal use of an LLM
# because the inputs are just two clean numbers — temperature and precipitation
# — and the output is one of three fixed labels. A simple rule-based approach
# like "temp between 10–25°C and precip < 1mm → good" would be faster, cheaper,
# and perfectly consistent across every run with no risk of unexpected output.
# The main thing you gain with an LLM is implicit domain knowledge — it can
# weigh edge cases like 9°C with wind differently without explicit rules — but
# for this narrow, well-defined task that flexibility doesn't justify the cost
# or the need for an "unknown" fallback. A rule-based approach would be the
# right engineering choice here.


import json
import os
from datetime import datetime
import requests
import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
ACCOUNT_URL = "https://manictd2026sa.blob.core.windows.net/"
CONTAINER = "pipeline-data"

TODAY = datetime.now().strftime("%Y-%m-%d")

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}

def get_blob_client(container: str, blob_name: str):
    credential = DefaultAzureCredential()
    service = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)
    return service.get_blob_client(container=container, blob=blob_name)


def reshape_hourly(data: dict) -> list[dict]:
    """Convert Open-Meteo parallel lists into per-hour record dicts."""
    hourly = data["hourly"]
    return [
        {
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        }
        for i in range(len(hourly["time"]))
    ]


def read_weather_records() -> list[dict]:
    blob_path = f"raw/{TODAY}/weather.json"
    print(f"Attempting to read blob: {blob_path}")

    try:
        client = get_blob_client(CONTAINER, blob_path)
        raw_bytes = client.download_blob().readall()
        data = json.loads(raw_bytes)
        records = reshape_hourly(data)
        print(f"Loaded {len(records)} records from Blob Storage.")
        return records

    except Exception:
        print(" Blob not found. Loading fallback from local file.")
        fallback_path = os.path.join(
            os.path.dirname(__file__), "..", "assignments_10", "resources", "weather_raw.json"
        )
        with open(fallback_path, "r") as f:
            data = json.load(f)
        records = reshape_hourly(data)
        print(f" Loaded {len(records)} records from fallback file.")
        return records



def classify_record(client: OpenAI, record: dict) -> str:
    """Call the LLM and return one of: good, marginal, bad, unknown."""
    user_msg = (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=5,
        temperature=0,
    )
    label = response.choices[0].message.content.strip().lower()
    return label if label in VALID_LABELS else "unknown"


def transform_records(records: list[dict]) -> list[dict]:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    subset = records[:24]
    enriched = []

    print(f"Classifying {len(subset)} records.")

    for i, record in enumerate(subset):
        label = classify_record(openai_client, record)
        enriched.append({**record, "conditions": label})

        if (i + 1) % 6 == 0:
            print(f"Progress: {i + 1}/{len(subset)} records classified.")

    print(f"Labels used: {set(r['conditions'] for r in enriched)}")
    return enriched


def write_enriched_records(enriched: list[dict]) -> None:
    blob_path = f"processed/{TODAY}/weather_classified.json"
    payload = json.dumps(enriched, indent=2).encode("utf-8")

    client = get_blob_client(CONTAINER, blob_path)
    client.upload_blob(payload, overwrite=True)
    print(f"Uploaded enriched data to: {blob_path}")



def spot_check() -> pd.DataFrame:
    blob_path = f"processed/{TODAY}/weather_classified.json"
    client = get_blob_client(CONTAINER, blob_path)
    raw_bytes = client.download_blob().readall()
    df = pd.DataFrame(json.loads(raw_bytes))

    print("\nconditions value_counts:")
    print(df["conditions"].value_counts().to_string())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    return df


def save_first_10(enriched: list[dict]) -> None:
    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    output_path = os.path.join(outputs_dir, "first_10_records.json")
    with open(output_path, "w") as f:
        json.dump(enriched[:10], f, indent=2)

    print(f"Saved first 10 records to: {output_path}")


if __name__ == "__main__":
    records = read_weather_records()
    enriched = transform_records(records)
    write_enriched_records(enriched)
    spot_check()
    save_first_10(enriched)

    print("\nPipeline complete.")
    
#Video Link
#https://youtu.be/RVFg8_whZCw