from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
"""
Concepts Question 1

Scenario A:
Best approach: RAG (Retrieval-Augmented Generation)

Reasoning:
The legal team's policy documents are large, frequently updated, and stored externally.
RAG is ideal because the system can retrieve the latest relevant policy chunks at query time
without retraining the model every quarter.

Scenario B:
Best approach: Fine-tuning

Reasoning:
The startup wants a consistent writing style that is unique and not common online.
Fine-tuning works well because they already have 3,000 examples of the exact tone and style
they want the model to learn permanently.

Scenario C:
Best approach: Prompt engineering

Reasoning:
The analyst only needs answers from a single short report for one-time use.
A carefully designed prompt with the report pasted into context is simpler and cheaper
than building a retrieval system or fine-tuning a model.
"""

"""
Concepts Question 2

A confidently wrong answer is more harmful because users are more likely to trust and act on it.
When a model sounds certain, people may not verify the information independently.

Example:
If a medical AI confidently gives the wrong medication dosage, a patient or doctor could make
a dangerous treatment decision that causes serious harm.

The tone matters because confidence affects perceived credibility. A response that says
"I am not sure" encourages caution and fact-checking, while a confident tone can create false trust
even when the content is incorrect.
"""

"""
Concepts Question 3

Correct RAG Pipeline Order

1. "Receive the user's query"
   The system first accepts the user's question or request.

2. "Embed the user's query"
   The query is converted into a numerical vector representation.

3. "Extract text from source documents"
   Raw text is pulled from PDFs, websites, databases, or other files.

4. "Split text into chunks"
   Large documents are divided into smaller searchable sections.

5. "Convert text chunks into embeddings"
   Each chunk is transformed into vector embeddings for similarity search.

6. "Retrieve the most relevant chunks"
   The system compares the query embedding to document embeddings and finds the best matches.

7. "Inject retrieved chunks into the prompt"
   The retrieved text is added into the prompt sent to the LLM.

8. "Generate a response from the LLM"
   The language model produces an answer using both the query and retrieved context.
"""

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}


# ---------------------------------------------------
# Keyword Question 1
# ---------------------------------------------------

query = "What are your hours on the weekend?"

result = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:", result[0][0])

"""
Explanation:
The selected document should be "hours.txt" because it contains overlapping keywords
such as "weekends" and information about opening and closing hours.
This is the most relevant match for the query.
"""


# ---------------------------------------------------
# Keyword Question 2
# ---------------------------------------------------

query = "Do you have anything without caffeine?"

result = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:", result[0][0])

"""
Explanation:

Selected document:
The system will likely select "menu.txt" because it contains beverage-related words
that overlap loosely with the query.

Did keyword RAG get this right?
Not really. The query asks about caffeine-free options, but the menu document does not
explicitly mention caffeine-free drinks. Keyword matching cannot understand semantic meaning,
so it only matches surface-level words.

What retrieval would do better?
Semantic/vector retrieval would work better because embeddings can capture meaning and recognize
that oat milk, almond milk, pastries, or other menu items may relate to caffeine-free options
even if the exact word "caffeine" is missing.
"""


# ---------------------------------------------------
# Keyword Question 3
# ---------------------------------------------------

"""
Prediction before running code:

I predict that "loyalty.txt" will be selected because the query mentions
"sign up" and "rewards," which are semantically related to a loyalty program.
The overlap may come from words like "rewards," "program," or similar concepts.
"""

query = "How do I sign up for rewards?"

result = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:", result[0][0])

"""
Result explanation:

The prediction was correct if "loyalty.txt" was selected.
Keyword retrieval succeeds here because the loyalty document contains related terms
such as "loyalty program" and "points," which overlap enough with the query.

If the result was surprising, it would likely be because keyword retrieval depends
entirely on exact token overlap and cannot truly understand synonyms or intent.
"""

"""
Semantic Question 1

What is a vector embedding?
A vector embedding is a numerical representation of text where words or passages with similar meanings
end up close together in mathematical space. Embeddings allow computers to compare meaning instead of
just matching exact words.

Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query.
Which chunk is more relevant?
The chunk with a similarity score of 0.85 is more relevant because its embedding is much closer to the
query embedding in vector space. A higher cosine similarity means the texts are more semantically related.

Why can semantic search find a relevant chunk even when none of the exact words appear?
Semantic search works because embeddings capture overall meaning and context, not just exact vocabulary.
Two passages can discuss the same concept using different wording, and their embeddings can still end up
near each other.
"""

"""
Semantic Question 2

| Feature                    | Keyword RAG                       | Semantic RAG                          |
|----------------------------|-----------------------------------|---------------------------------------|
| What is compared?          | Exact word overlap                | Vector embeddings / semantic meaning  |
| What is retrieved?         | Full document                     | Relevant text chunks                  |
| Can it handle synonyms?    | No                                | Yes                                   |
| Storage format             | Plain text dictionary             | Vector database / embedding index     |
| Relevance score            | Number of overlapping keywords    | Cosine similarity score               |
"""


# ---------------------------------------------------
# LlamaIndex Question 1
# ---------------------------------------------------

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Load BrightLeaf PDFs
documents = SimpleDirectoryReader(
    "../../06_AI_augmentation/brightleaf_pdfs"
).load_data()

# Build index
index = VectorStoreIndex.from_documents(documents)

# Query engine
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print("\n" + "=" * 70)
    print("QUESTION:")
    print(q)

    response = query_engine.query(q)

    print("\nANSWER:")
    print(response)

    print("\nSOURCE NODES:")
    for i, node in enumerate(response.source_nodes, start=1):
        print(f"\nNode {i}")
        print(f"Score: {node.score:.4f}")
        print(node.text[:150])


"""
Query 1 observations:
The retrieved chunks should mostly discuss HR policies, compensation, healthcare,
vacation, or other employee benefits. These chunks are likely highly relevant.

The model response will probably sound fairly confident and specific because the
documents likely contain direct answers.

Unexpected retrievals might include unrelated HR or policy sections if embeddings
match on broad workplace terminology.
"""

"""
Query 2 observations:
The retrieved chunks should include cybersecurity rules, password policies,
data handling, or internal access procedures.

The model may hedge slightly if the security information is spread across multiple
documents or only partially covered.

Unexpected chunks may appear if words like "security" overlap with financial or
physical safety contexts.
"""


# ---------------------------------------------------
# LlamaIndex Question 2
# ---------------------------------------------------

query = "What employee benefits does BrightLeaf offer?"

# top_k = 1
engine_k1 = index.as_query_engine(similarity_top_k=1)
response_k1 = engine_k1.query(query)

print("\n" + "=" * 70)
print("TOP K = 1")
print(response_k1)

for node in response_k1.source_nodes:
    print(f"Score: {node.score:.4f}")
    print(node.text[:150])


# top_k = 5
engine_k5 = index.as_query_engine(similarity_top_k=5)
response_k5 = engine_k5.query(query)

print("\n" + "=" * 70)
print("TOP K = 5")
print(response_k5)

for node in response_k5.source_nodes:
    print(f"Score: {node.score:.4f}")
    print(node.text[:150])


"""
Observations:
With top_k=1, the response may be shorter and rely heavily on a single chunk.
With top_k=5, the response may include more detail because additional context is available.

More retrieved context is not always better. Too many chunks can introduce noise,
irrelevant details, or conflicting information that makes the final response less focused.
"""


# ---------------------------------------------------
# LlamaIndex Question 3
# ---------------------------------------------------

challenging_query = "What partnerships does BrightLeaf have with international governments?"

response = query_engine.query(challenging_query)

print("\n" + "=" * 70)
print("CHALLENGING QUERY")
print(challenging_query)

print("\nANSWER:")
print(response)

print("\nSOURCE NODES:")
for node in response.source_nodes:
    print(f"Score: {node.score:.4f}")
    print(node.text[:300])


"""
Expected result:
I expected the pipeline to struggle because the documents may not contain information
about international government partnerships.

What actually happened:
The system may still retrieve vaguely related business or policy chunks because semantic
search tries to find the closest available meaning even when the exact answer is absent.

What I would improve:
I would add better retrieval filtering, confidence thresholds, or a fallback mechanism
that tells the user when the answer is not supported by the documents.
"""


# ---------------------------------------------------
# LlamaIndex Question 4
# ---------------------------------------------------

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

judge_llm = OpenAI(model="gpt-4o-mini")

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

# Good query
q1 = "What employee benefits does BrightLeaf offer?"
response1 = query_engine.query(q1)

faithfulness_result1 = faithfulness_evaluator.evaluate_response(
    query=q1,
    response=response1,
)

relevancy_result1 = relevancy_evaluator.evaluate_response(
    query=q1,
    response=response1,
)

print("\n" + "=" * 70)
print("GOOD QUERY EVALUATION")

print("Faithfulness Score:", faithfulness_result1.passing)
print("Relevancy Score:", relevancy_result1.passing)


# Poor query
q2 = "What is BrightLeaf's office cafeteria menu?"
response2 = query_engine.query(q2)

faithfulness_result2 = faithfulness_evaluator.evaluate_response(
    query=q2,
    response=response2,
)

relevancy_result2 = relevancy_evaluator.evaluate_response(
    query=q2,
    response=response2,
)

print("\n" + "=" * 70)
print("POOR QUERY EVALUATION")

print("Faithfulness Score:", faithfulness_result2.passing)
print("Relevancy Score:", relevancy_result2.passing)


"""
Evaluation Concepts

What does a faithfulness score of 1.0 mean?
A faithfulness score of 1.0 means the response is fully supported by the retrieved
source documents and does not appear to invent information.

What would a score of 0.0 indicate?
A score of 0.0 would indicate hallucination or unsupported claims that are not backed
by the retrieved context.

What does a relevancy score measure?
Relevancy measures whether the response actually answers the user's question.
A response can be faithful to the documents but still irrelevant to the question asked.

How is relevancy different from faithfulness?
Faithfulness checks grounding in the sources, while relevancy checks usefulness and
alignment with the user's query.

Did the scores change between queries?
The unsupported cafeteria-menu query will likely receive lower relevancy and possibly
lower faithfulness scores because the documents probably do not contain that information.

What is the "LLM-as-a-judge" approach?
LLM-as-a-judge uses another language model to evaluate the quality of generated answers.
It is useful for RAG systems because many responses are subjective, open-ended, or difficult
to measure with a simple right-or-wrong accuracy metric.
"""