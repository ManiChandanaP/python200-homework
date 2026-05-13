from pathlib import Path
from dotenv import load_dotenv
import os

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)

# Step 1: Setup
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "OPENAI_API_KEY not found in .env file"
print("API key loaded successfully.")
docs_dir = Path("assignments_06/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
print(f"Document directory found: {docs_dir}")

# Step 2: Load the Documents


documents = SimpleDirectoryReader(str(docs_dir)).load_data()
print(f"\nLoaded {len(documents)} documents.\n")
for doc in documents:
    file_name = doc.metadata.get("file_name", "Unknown file")
    print(f"- {file_name}")

# Step 3: Build the Index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
print("\nIndex built successfully. Ready to answer questions.\n")

# Step 4: Query the Assistant


questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:

    print("=" * 80)
    print(f"QUESTION:\n{question}\n")

    response = query_engine.query(question)

    print("ANSWER:")
    print(response)
    print()

    # Retrieve top source node
    top_node = response.source_nodes[0]

    file_name = top_node.node.metadata.get("file_name", "Unknown file")
    similarity_score = top_node.score
    chunk_text = top_node.node.text[:200]

    print("TOP RETRIEVED SOURCE:")
    print(f"Document: {file_name}")
    print(f"Similarity Score: {similarity_score:.4f}")
    print(f"Chunk Preview:\n{chunk_text}")
    print()


# Reflection:
# The assistant generally sounded confident and accurate.
# The answers tied closely to the retrieved documents, especially
# for factual questions like hours and catering services.
# One interesting observation was how smoothly the model summarized
# information from longer narrative documents like the company history.
# The retrieval step clearly helped ground the responses in the source text.


# Step 5: Find a Failure


failure_question = (
    "Which menu item is the most popular with customers?"
)

print("\n" + "=" * 80)
print("FAILURE TEST")
print("=" * 80)

print(f"\nQUESTION:\n{failure_question}\n")

failure_response = query_engine.query(failure_question)

print("FULL RESPONSE:")
print(failure_response)
print()

print("ALL RETRIEVED SOURCE NODES:\n")

for i, node in enumerate(failure_response.source_nodes, start=1):

    file_name = node.node.metadata.get("file_name", "Unknown file")
    similarity_score = node.score
    chunk_text = node.node.text[:200]

    print(f"Source Node {i}")
    print(f"Document: {file_name}")
    print(f"Similarity Score: {similarity_score:.4f}")
    print(f"Chunk Preview:\n{chunk_text}")
    print("-" * 60)


# Failure 

# I asked which menu item is the most popular with customers.
# I expected this to be difficult because popularity rankings
# were probably not included anywhere in the documents.
# The retrieval system still returned somewhat related menu documents,
# but none of them actually answered the question directly.
# Depending on the model behavior, it may still try to guess based on
# the menu content instead of admitting that the information is missing.
# This demonstrates an important limitation of RAG:
# retrieval can provide relevant context, but the language model may
# still generate unsupported conclusions.
# The tone often remains confident even when the answer is uncertain,
# which suggests users should not automatically trust fluent AI responses.

# To improve the system, I would:
# - Add stronger prompting instructions telling the model to say
#   "I don't know" when evidence is missing
# - Use metadata filtering or confidence thresholds
# - Add citation requirements for generated answers
# - Evaluate retrieval quality systematically


# Step 6: Reflection

"""
REFLECTION

1. Manual semantic RAG vs. LlamaIndex

The manual semantic RAG pipeline from the lesson required many lines
of code for chunking, embedding generation, vector storage, similarity
search, and retrieval orchestration.

Using LlamaIndex dramatically reduced the implementation size.
The equivalent workflow in this project took roughly 10–20 lines
for the core indexing and querying logic.

This demonstrates the value of frameworks:
they abstract away repetitive infrastructure code so developers can
focus on application behavior and experimentation instead of rebuilding
retrieval systems from scratch.


2. Another valuable business use case

A strong real-world use case would be an HR policy assistant for a company.

Employees could ask questions like:
- "How many vacation days do I get?"
- "What is the parental leave policy?"
- "How do I enroll in benefits?"

Instead of searching through long HR documents manually,
the assistant could retrieve the relevant policy sections instantly.
This would save time for both employees and HR staff.


3. One failure mode RAG cannot fully prevent

Even when retrieval works correctly, the language model can still
misinterpret retrieved information or combine details incorrectly.

For example, if multiple documents contain slightly conflicting
information, the model may synthesize an answer that sounds coherent
but is inaccurate.

RAG improves factual grounding, but it does not eliminate hallucinations
or reasoning mistakes entirely.
"""