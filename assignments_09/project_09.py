import json
import os
import requests
import pandas as pd
from datetime import date
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

ACCOUNT_URL = "https://manictd2026sa.blob.core.windows.net/"
CONTAINER = "pipeline-data"
OUTPUTS_DIR = Path(__file__).parent / "outputs"


# STEP 1: EXTRACT
print("STEP 1: Extracting weather data from Open-Meteo.")

url = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=35.2271"
    "&longitude=-80.8431"
    "&hourly=temperature_2m,precipitation"
    "&forecast_days=7"
)

response = requests.get(url)
response.raise_for_status()

weather_data = response.json()
print(f" Received data for: {weather_data.get('timezone', 'Unknown location')}")
print(f" Hours of data: {len(weather_data['hourly']['time'])}")

# STEP 2: SERIALIZE

print("\nSTEP 2: Serializing data to JSON bytes.")

json_bytes = json.dumps(weather_data).encode("utf-8")
print(f" Serialized size: {len(json_bytes):,} bytes")


# STEP 3: LOAD — Upload to Azure Blob Storage

print("\nSTEP 3: Uploading to Azure Blob Storage.")

today = date.today().isoformat()
blob_path = f"raw/{today}/weather.json"

blob_service_client = BlobServiceClient(
    account_url=ACCOUNT_URL,
    credential=DefaultAzureCredential()
)
blob_client = blob_service_client.get_blob_client(
    container=CONTAINER,
    blob=blob_path
)

blob_client.upload_blob(json_bytes, overwrite=True)

print(f" Uploaded to blob path : {blob_path}")
print(f" Bytes uploaded        : {len(json_bytes):,}")


# STEP 4: VERIFY — List all blobs
print("\nSTEP 4: Verifying — listing all blobs in container...")

container_client = blob_service_client.get_container_client(CONTAINER)
blobs = list(container_client.list_blobs())

if blobs:
    print(f"  Found {len(blobs)} blob(s):")
    for blob in blobs:
        print(f" {blob.name}  ({blob.size:,} bytes)")
else:
    print("  No blobs found (something may have gone wrong).")

# STEP 5: READ BACK
print("\nSTEP 5: Reading back the uploaded blob.")

downloaded = blob_client.download_blob().readall()
parsed = json.loads(downloaded)
# Load the "hourly" section into a DataFrame
df = pd.DataFrame(parsed["hourly"])
print("\n  First 5 rows of hourly weather data:")
print(df.head().to_string(index=False))

output_file = OUTPUTS_DIR / "weather_raw.json"
output_file.write_bytes(downloaded)
print(f"\n Saved raw JSON locally to: {output_file}")
print("Pipeline complete!")