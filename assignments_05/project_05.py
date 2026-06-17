# --- Setup ---
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()


# --- Helper Function ---
def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


# --- System Prompt ---
SYSTEM_PROMPT = """
You are a professional job application coach helping people transition into new careers.

Guidelines:
- Stay focused only on resumes, cover letters, and job applications
- Give practical, clear, and actionable advice
- Do NOT invent skills or experience that the user did not mention
- Always remind the user to review and edit your output before submitting anywhere
- Acknowledge that hiring expectations vary by industry and that the user should use
  their own judgment when applying your suggestions
"""

# Comment:
# I made the prompt strict about "not inventing experience" to prevent hallucinations,
# which is critical for job applications where accuracy matters.


#Task 2: Bullet Rewriter
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format bullets into a delimited block so the model clearly separates
    # instructions from user content — reduces prompt injection risk and improves accuracy.
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
You are a professional resume coach helping a career changer.

Rewrite each resume bullet point below to be:
- Specific and results-oriented
- Starting with a strong action verb
- Impactful without inventing facts not implied by the original

Respond ONLY with valid JSON — no preamble, no explanation, no markdown fences.

Format exactly like this:
[
  {{"original": "...", "improved": "..."}}
]

Bullet points:
```
{bullet_text}
```
"""
    # The delimiters (```) clearly separate the bullet content from the instructions above,
    # which prevents the model from treating user content as additional directives.

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages)

    try:
        data = json.loads(response)

        print("\nRewritten Bullets")
        for item in data:
            print(f"\nOriginal: {item['original']}")
            print(f"Improved: {item['improved']}")

        return data

    except json.JSONDecodeError:
        cleaned = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(cleaned)
            print("\nRewritten Bullets")
            for item in data:
                print(f"\nOriginal: {item['original']}")
                print(f"Improved: {item['improved']}")
            return data
        except json.JSONDecodeError:
            print("JSON parsing failed. Raw response:")
            print(response)
            return []


# Test bullets
# Comment:
# These bullets are weak because they are vague, use passive/generic verbs ("helped",
# "made", "worked"), and have no measurable results or context.
# The model improves them by adding strong action verbs (e.g. "Resolved", "Produced"),
# specifying the scope of work, and implying or surfacing results where possible.
test_bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

print("\nTask 2 Test")
rewrite_bullets(test_bullets)


# Task 3: Cover Letter Generator
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
You write strong cover letter opening paragraphs for career changers.

The paragraph should be:
- 3-5 sentences
- Confident and specific
- Free of generic clichés like "I am excited to bring my unique skills"
- Tailored to the specific role and background provided

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening: After seven years as a registered nurse, I've spent my career making decisions
under pressure using incomplete information — which turns out to be excellent training for
data analysis. I recently completed a data analytics program where I built dashboards
tracking patient outcomes across departments. I'm excited to bring that combination of
clinical context and technical skill to [Company]'s mission-driven work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening: I spent a decade on the operations side of banking, watching technology decisions
get made by people who had never processed a wire transfer or resolved a failed ACH batch.
That frustration turned into curiosity, and two years of self-teaching Python later, I'm
ready to be on the other side of those decisions. I'm applying to [Company] because your
work on payment infrastructure is exactly where my domain expertise and new technical skills
intersect.

Now write one:

Role: {job_title}
Background: {background}
Opening:
"""

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages)


# Comment:
# I chose examples from career changers specifically (nurse→data, banking→engineering)
# because that mirrors the target user. The few-shot pattern controls tone (confident,
# narrative-driven) and prevents the model from producing generic openings.
# Both examples show the "reframe past experience as an asset" structure I want replicated.

# Test cover letter
print("\nTask 3 Test")
test_job_title = "Junior Data Engineer"
test_background = (
    "Five years of experience as a middle school math teacher; "
    "recently completed a Python course and built data pipelines using Prefect and Pandas."
)
cover_letter = generate_cover_letter(test_job_title, test_background)
print("\nCover Letter")
print(cover_letter)


#Task 4: Moderation
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print("\nYour input was flagged by our content filter. Please rephrase and try again.\n")
        return False

    return True


# Test moderation
print("\nTask 4: Moderation Tests")
print("Safe input result   :", is_safe("Can you help me improve my resume?"))
print("Flagged input result:", is_safe("I want to harm someone"))


# Task 5: Chatbot
def run_chatbot():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    print("Job Application Helper")
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # Skip empty input
        if not user_input:
            continue

        # Run moderation check before doing anything else
        if not is_safe(user_input):
            continue

        # Bullet rewriting
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")

            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            if raw_bullets:
                rewrite_bullets(raw_bullets)
            else:
                print("No bullets entered. Please try again.")

        # Cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()

            if job_title and background:
                result = generate_cover_letter(job_title, background)
                print("\nCover Letter Opening")
                print(result)
                print("\n(Remember to review and personalize this before submitting.)")
            else:
                print("Job Application Helper: Please provide both a job title and background.")

        else:
            messages.append({"role": "user", "content": user_input})

            reply = get_completion(messages)

            print(f"\nJob Application Helper: {reply}\n")

            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_chatbot()


#Task 6: Ethics Reflection (Option A)
"""
Option A — Comment block

Question 1: Bias in training data
The model was trained on text produced predominantly by English-speaking professionals in
Western corporate contexts. This means its suggestions may favor communication styles common
in industries like tech, finance, or consulting — using vocabulary, tone, and achievement-framing
that can feel foreign or unnatural to candidates from different cultural backgrounds or
non-corporate fields like education or healthcare. A teacher rewriting bullets might get
suggestions that sound less like education and more like management consulting, which could
actually hurt their application in some contexts. Users should treat output as a starting
point and localize the language to match their actual industry norms.

Question 2: Risks of submitting output without review
If a job-seeker submits AI output without reviewing it, they risk including inaccurate
statements — for example, bullet points that subtly overstate scope or cover letters that
reference a company detail the model hallucinated. Beyond factual risk, unreviewed output
often reads as generic, which experienced hiring managers can detect. In worst cases,
discrepancies between the resume and what a candidate actually says in an interview can
create serious credibility problems. The assistant explicitly tells users to review output
before submitting, but a stronger guardrail would be a UI that requires an explicit
"I have reviewed this" confirmation before allowing copy.
"""