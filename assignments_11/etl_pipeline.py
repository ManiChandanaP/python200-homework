# Video link: https://youtu.be/x6_j5mzzRQY


import json
import os
from datetime import date

import requests
from dotenv import load_dotenv
from openai import OpenAI
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from prefect import flow, task
from prefect.logging import get_run_logger

load_dotenv()
# Configuration

CITY = "Charlotte, NC"
LATITUDE = 35.2271
LONGITUDE = -80.8431
STORAGE_ACCOUNT = "https://manictd2026sa.blob.core.windows.net/"
CONTAINER = "pipeline-data"


# Extract

@task(retries=2, retry_delay_seconds=10)
def extract_weather() -> dict:
    logger = get_run_logger()
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LATITUDE}&longitude={LONGITUDE}"
        "&hourly=temperature_2m,precipitation"
        "&forecast_days=7"
    )
    logger.info(f"Fetching weather data for {CITY}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    print(f"[extract] Fetched {len(data['hourly']['time'])} hourly records from Open-Meteo.")
    return data



# Transform

@task
def transform_and_classify(raw: dict) -> list:
    logger = get_run_logger()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    times = raw["hourly"]["time"]
    temps = raw["hourly"]["temperature_2m"]
    precip = raw["hourly"]["precipitation"]

    records = [
        {"time": t, "temperature_c": temp, "precipitation_mm": p}
        for t, temp, p in zip(times, temps, precip)
    ]


    logger.info(f"Classifying first 24 of {len(records)} records...")
    valid_labels = {"good", "marginal", "bad"}

    for i, record in enumerate(records[:24]):
        prompt = (
            f"Temperature: {record['temperature_c']}°C, "
            f"Precipitation: {record['precipitation_mm']}mm"
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are classifying hourly weather conditions for outdoor running. "
                            "Given a temperature in Celsius and a precipitation amount in mm, "
                            "classify the conditions as exactly one of: good, marginal, or bad. "
                            "Reply with that one word only -- no punctuation, no explanation."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=5,
                temperature=0,
            )
            label = response.choices[0].message.content.strip().lower()
            record["running_condition"] = label if label in valid_labels else "unknown"
        except Exception as e:
            logger.warning(f"Classification failed for record {i}: {e}")
            record["running_condition"] = "unknown"

        if (i + 1) % 6 == 0:
            print(f"[transform] Classified {i + 1}/24 records...")

   
    for record in records[24:]:
        record["running_condition"] = None

    print(f"[transform] Done. {len(records)} total records, 24 classified.")
    return records



# Load

@task
def load_to_blob(records: list) -> str:
    logger = get_run_logger()
    today = date.today().isoformat()
    blob_path = f"final/{today}/weather_etl.json"

    credential = DefaultAzureCredential()
    client = BlobServiceClient(account_url=STORAGE_ACCOUNT, credential=credential)
    blob_client = client.get_blob_client(container=CONTAINER, blob=blob_path)

    payload = json.dumps(records, indent=2).encode("utf-8")
    blob_client.upload_blob(payload, overwrite=True)

    logger.info(f"Uploaded {len(payload):,} bytes to {blob_path}")
    print(f"[load] Uploaded to {CONTAINER}/{blob_path} ({len(payload):,} bytes)")
    return blob_path

# Flow

@flow(log_prints=True)
def weather_etl_pipeline():
    raw = extract_weather()
    records = transform_and_classify(raw)
    blob_path = load_to_blob(records)
    print(f"[flow] Pipeline complete. Results at: pipeline-data/{blob_path}")


if __name__ == "__main__":
    weather_etl_pipeline()