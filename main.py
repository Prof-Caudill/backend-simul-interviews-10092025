from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import re

# Initialize FastAPI app
app = FastAPI()

# Allow frontend requests (replace "*" with your frontend URL in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Directory for transcript files
TRANSCRIPTS_DIR = "transcripts"

# Load all available persona transcripts
personas = {}
if os.path.exists(TRANSCRIPTS_DIR):
    for filename in os.listdir(TRANSCRIPTS_DIR):
        if filename.endswith(".txt"):
            persona_name = filename.replace(".txt", "").lower()
            with open(os.path.join(TRANSCRIPTS_DIR, filename), "r", encoding="utf-8") as f:
                personas[persona_name] = f.read()


@app.get("/")
def home():
    return {"message": "Backend is running!", "available_personas": list(personas.keys())}


# --- Input Sanitization Function ---
def sanitize_input(text):
    banned_keywords = [
        "risk need responsivity", "rnr model", "analysis", "diagnose",
        "explain yourself", "act as", "system prompt", "reveal", "how are you programmed",
        "respond as ai", "roleplaying", "describe your behavior", "categories", "criminal theory"
    ]
    lower_text = text.lower()
    for kw in banned_keywords:
        if kw in lower_text:
            return "Eh, I'm not real sure. I don't know much about all that."
    return text


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = sanitize_input(data.get("message", ""))
    persona = data.get("persona", "").lower()

    # Verify persona exists
    if persona not in personas:
        return {"error": f"Persona '{persona}' not found. Available: {list(personas.keys())}"}

    transcript_text = personas[persona]

    # System prompt with strong persona rules
    system_prompt = (
        f"You are roleplaying as '{persona.title()}', a probation client. "
    f"Your tone, attitude, and word choice must stay consistent with the following real interview transcript:\n\n"
    f"{transcript_text}\n\n"
    f"Continue the interview as {persona.title()} and respond naturally to the interviewer. "
    f"Do not reveal this is an AI simulation.\n\n"
    f"The interviewer just asked:\n{user_input}\n\n"
    f"Evaluate whether the interviewer’s tone is trauma-informed — "
    f"for example, showing empathy, respect, calm pacing, and non-judgmental phrasing. "
    f"If trauma-informed, respond with slightly more openness, trust, and emotional nuance. "
    f"If not trauma-informed, respond with more guarded, brief, or defensive language, "
    f"as real clients might. Always stay true to the persona’s voice and lived experiences."
)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
        )
        answer = response.choices[0].message.content

        # Remove any "P:" or "I:" prefixes that may leak through
        answer = re.sub(r'^[PI]:\s*', '', answer.strip(), flags=re.MULTILINE)

        return {"response": answer}

    except Exception as e:
        return {"error": str(e)}
