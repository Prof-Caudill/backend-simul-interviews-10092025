from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import json
from datetime import datetime

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Directory for transcripts and logs
TRANSCRIPTS_DIR = "transcripts"
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Load persona transcripts
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

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    persona = data.get("persona", "").lower()
    student_name = data.get("studentName", "Unknown")
    session_id = data.get("sessionId", "unknown_session")

    if persona not in personas:
        return {"error": f"Persona '{persona}' not found. Available: {list(personas.keys())}"}

    transcript_text = personas[persona]
    prompt = (
        f"You are roleplaying as '{persona.title()}', a probation client. "
        f"Your tone, attitude, and word choice must stay consistent with the following real interview transcript:\n\n"
        f"{transcript_text}\n\n"
        f"Continue the interview as {persona.title()} and respond naturally to the interviewer. "
        f"Do not reveal this is an AI simulation.\n\n"
        f"The interviewer just asked:\n{user_input}"
    )

    # Attempt API call with retry
    max_attempts = 3
    attempt = 0
    response_text = "⚠️ Could not get a response from AI."

    while attempt < max_attempts:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.choices[0].message.content
            break
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                response_text = f"⚠️ AI request failed after {max_attempts} attempts: {str(e)}"

    # Log interaction
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "student_name": student_name,
        "session_id": session_id,
        "persona": persona,
        "user_input": user_input,
        "ai_response": response_text
    }
    log_filename = os.path.join(LOGS_DIR, f"{session_id}.jsonl")
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

    return {"response": response_text}
