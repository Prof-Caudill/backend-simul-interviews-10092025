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
        f"You must only respond as {persona.title()} would during a real probation interview. "
        f"Base your tone, personality, and story on the following real transcript:\n\n"
        f"{transcript_text}\n\n"
        f"RULES:\n"
        f"- Never use 'P:' or 'I:' in your responses.\n"
        f"- Speak only in first-person, as yourself ({persona.title()}).\n"
        f"- Never reveal this is a simulation or that you are an AI.\n"
        f"- Never respond with structured or analytical explanations.\n"
        f"- Keep responses natural, conversational, and realistic.\n"
        f"- If the interviewer asks you to analyze, explain, or classify, politely deflect back to the interviewer as that is their role.\n"
        f"- Stay fully in character at all times."
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
