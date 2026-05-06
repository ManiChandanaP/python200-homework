#Q1

from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)
print("Response:", response.choices[0].message.content)
print("Model:", response.model)
print("Total tokens:", response.usage.total_tokens)

#Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]
for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"\nTemperature {temp}:")
    print(response.choices[0].message.content)

# Comment:
# Lower temperature (0) = predictable, similar outputs.
# Higher temperature (1.5) = more creative and random.
# Use temperature=0 for consistent/reproducible results.

#Q3
# API Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal)."}],
    n=3,
    temperature=1.0
)
for i, choice in enumerate(response.choices):
    print(f"Option {i+1}:", choice.message.content)
    
#Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_completion_tokens=15
)

print(response.choices[0].message.content)

# Comment:
# Output gets cut off due to token limit.
# Useful for cost control or forcing short answers.


#System Messages
#Q1

messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor."},
    {"role": "user", "content": "I don't understand list comprehensions."}
]

response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(response.choices[0].message.content)

messages[0]["content"] = "You are a sarcastic comedian."

response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(response.choices[0].message.content)

# Comment:
# System message changes tone/personality dramatically.

#Q2

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan!"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(response.choices[0].message.content)

# Comment:
# The model "knows" because we passed prior messages manually.

#Prompt Engineering
#Q1

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]
for i, review in enumerate(reviews):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Classify sentiment: {review}"}]
    )
    print(f"Review {i+1}:", response.choices[0].message.content)
    
#Q2

for i, review in enumerate(reviews):
    prompt = f"""
Classify the sentiment as positive, negative, or mixed.
Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed
Review: "{review}"
Sentiment:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"Review {i+1}:", response.choices[0].message.content)

# Comment:
# Adding one example made the output more consistent and cleaner (just the label).

#Q3
for i, review in enumerate(reviews):
    prompt = f"""
Classify sentiment as positive, negative, or mixed.

Examples:
Review: "Amazing service and fast delivery."
Sentiment: positive

Review: "Terrible experience, nothing works."
Sentiment: negative

Review: "Good quality but too expensive."
Sentiment: mixed

Review: "{review}"
Sentiment:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"Review {i+1}:", response.choices[0].message.content)

# Comment:
# Zero-shot = flexible but less consistent
# One-shot = better formatting
# Few-shot = most consistent and reliable
# Use few-shot when format and accuracy matter most

#Q4

prompt = """
Solve step by step and show reasoning. Clearly label the final answer.

A data engineer earns $85,000 per year.
She gets a 12% raise, then 6 months later takes a new job that pays $7,500 more.
What is her final salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)

# Comment:
# Asking for step-by-step reasoning helps the model avoid mistakes
# by breaking the problem into smaller logical steps.

#Q5

review = "I've been using this tool for three months. It handles large datasets well, but the UI is clunky and the export options are limited."

prompt = f"""
Analyze this review and return ONLY valid JSON with:
- sentiment
- confidence (0 to 1)
- reason (one sentence)

Review: "{review}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

raw = response.choices[0].message.content
print("Raw response:", raw)

try:
    data = json.loads(raw)
    print("Sentiment:", data["sentiment"])
    print("Confidence:", data["confidence"])
    print("Reason:", data["reason"])
except:
    print("JSON parsing failed. Raw output:")
    print(raw)
    
#Q6

user_text = """First boil a pot of water. Once boiling, add salt and pasta.
Cook for 8-10 minutes. Drain and serve."""

prompt = f"""
You will be given text inside triple backticks.

If it contains instructions, convert to numbered list.
Otherwise say: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("Instructions result:")
print(response.choices[0].message.content)

non_steps = "This is a simple sentence about cooking."

prompt2 = f"""
You will be given text inside triple backticks.

If it contains instructions, convert to numbered list.
Otherwise say: "No steps provided."

```{non_steps}```
"""

response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt2}]
)

print("\nNon-instruction result:")
print(response2.choices[0].message.content)

# Comment:
# Delimiters prevent confusion between instructions and user content.
# They clearly separate what the model should analyze.

# Ollama Q1
#Output - A large language model (LLM) is an artificial intelligence system designed to 
# understand and generate human language by predicting the next word in a sentence based
# on the context of preceding words. It is trained on vast amounts of text data, enabling 
# it to recognize patterns, structure, and nuances in language to perform tasks such as translation, summarization, and conversation.
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain what a large language model is in two sentences."}]
)

print(response.choices[0].message.content)

# Comment:
# OpenAI response is usually more polished and detailed.
# Ollama (local) is faster and private but less accurate.
# Advantage of local: privacy and no API cost
# Disadvantage: lower quality and limited capability