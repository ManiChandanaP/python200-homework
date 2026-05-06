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
- Always remind the user to review and edit your output before submitting
- Acknowledge that hiring expectations vary by industry and the user should use their own judgment
"""

# Comment:
# I made the prompt strict about "not inventing experience" to avoid hallucinations,
# which is critical for job applications where accuracy matters.


# --- Task 2: Bullet Rewriter ---
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
You are a professional resume coach helping a career changer.

Rewrite each resume bullet point to be:
- Specific
- Results-oriented
- Strong and impactful

Do NOT invent new facts.

Respond ONLY with valid JSON (no extra text).

Format:
[
  {{"original": "...", "improved": "..."}}
]

Bullet points:"""

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages)

    try:
        data = json.loads(response)

        print("\n--- Rewritten Bullets ---")
        for item in data:
            print("\nOriginal:", item["original"])
            print("Improved:", item["improved"])

        return data

    except:
        print("JSON parsing failed. Raw response:")
        print(response)
        return []


# Test bullets
# Comment:
# These bullets are weak because they are vague and lack measurable impact.
# The model improves them by adding action verbs and making them more specific.
test_bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]


# --- Task 3: Cover Letter Generator ---
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
You write strong cover letter opening paragraphs for career changers.

The paragraph should be:
- 3-5 sentences
- Confident and specific
- Avoid generic phrases

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
# I chose examples from career changers to guide tone and structure.
# Few-shot prompting helps control style and avoid generic outputs.


# --- Task 4: Moderation ---
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print("\n Please rephrase your input — it may violate content guidelines.\n")
        return False

    return True


# Test moderation
print("\n--- Moderation Tests ---")
print("Safe test:", is_safe("Can you help improve my resume?"))
print("Flagged test:", is_safe("I want to harm someone"))


# --- Task 5: Chatbot ---
def run_chatbot():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("1. Resume bullet rewriting")
    print("2. Cover letter drafting")
    print("3. General application advice")
    print("\nType 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("\nGood luck with your applications!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        # Bullet rewriting
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nPaste your bullet points (type DONE when finished):")

            bullets = []
            while True:
                line = input()
                if line.strip().upper() == "DONE":
                    break
                if line.strip():
                    bullets.append(line)

            rewrite_bullets(bullets)

        # Cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job title: ").strip()
            background = input("Your background: ").strip()

            result = generate_cover_letter(job_title, background)

            print("\n--- Cover Letter ---")
            print(result)

        # Regular chat
        else:
            messages.append({"role": "user", "content": user_input})

            reply = get_completion(messages)

            print("\nAssistant:", reply)

            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_chatbot()


# --- Task 6: Ethics Reflection ---
"""
If a job-seeker submits AI-generated content without reviewing it, they risk including
inaccurate or generic statements that could harm their chances or misrepresent their experience.

One important guardrail would be requiring users to review and confirm edits before copying output,
along with a visible disclaimer reminding them that the AI may not fully understand their field.
"""