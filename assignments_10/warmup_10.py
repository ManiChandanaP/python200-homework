#Q1

# Parse "Jan 5th, 2024" → "2024-01-05": Use deterministic code; dateutil.parser handles this exactly with no ambiguity.
# Classify "my card was charged twice": Use an LLM; freeform text requires language understanding to determine intent.
# Calculate average of a list: Use deterministic code; this is pure math, no language understanding needed.
# Extract company from "Sr. Data Eng @ Acme Corp (contract)": Use an LLM; format varies too much for reliable regex.
# Check if review > 100 words: Use deterministic code; len(text.split()) > 100 is exact and instant.

#Q2

# Problem: The prompt returns free-form prose every time. Downstream code
# can't reliably parse it — no consistent structure, no fixed keys to extract.
#
# Rewritten prompt:
# system = """
# Analyze the product review and return ONLY a JSON object with these keys:
#   - "summary": 1-2 sentence summary
#   - "sentiment": one of "positive", "neutral", or "negative"
#   - "key_issue": main topic in 5 words or fewer
# No extra text. No markdown. Valid JSON only.
# """

#Q3
# 1. 50,000 records × 1 second = 50,000 seconds = ~13.9 hours. Not practical.
#
# 2. Use the OpenAI Batch API — submit all 50,000 requests as one JSONL file.
#    OpenAI processes them async within 24 hours at 50% lower cost, with no
#    rate-limit pressure per request.

#Azure Q1

# 1. Data residency/compliance — Azure keeps data in your chosen region,
#    satisfying GDPR, HIPAA, or FedRAMP. OpenAI's API has no such guarantee.
#
# 2. Private networking — the endpoint can be locked inside a VNet with
#    Private Link so traffic never touches the public internet. OpenAI's
#    API doesn't support this.

#Q2
# The three Azure-specific parameters (not api_key):
#
# 1. azure_endpoint — the full URL of your Azure OpenAI resource.
#    e.g. "https://my-resource.openai.azure.com/"
#
# 2. api_version — the REST API version date string Azure uses to version its API.
#    e.g. "2024-02-01"
#
# 3. azure_deployment — the custom name YOU gave the deployment in Azure OpenAI Studio.
#    This replaces the model name in API calls.

#Q3

# Instead of a model name like "gpt-4o-mini", you pass your DEPLOYMENT NAME —
# the custom label you gave it when deploying in Azure OpenAI Studio.
# e.g. if you named it "gpt4o-mini-prod", you pass model="gpt4o-mini-prod".
#
# Where to find it:
# Azure Portal → your OpenAI resource → Azure OpenAI Studio → Deployments tab
# → copy the value in the "Deployment name" column.