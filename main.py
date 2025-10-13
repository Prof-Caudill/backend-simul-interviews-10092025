from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from datetime import datetime

# Initialize FastAPI
app = FastAPI()

# Allow frontend connections (adjust "*" in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Directories
TRANSCRIPTS_DIR = "transcripts"
LOGS_DIR = "logs"

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Load available persona transcripts
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
    student_name = data.get("student_name", "").strip()  # NEW: capture student name

    if not student_name:
        return {"error": "Missing student name."}

    if persona not in personas:
        return {"error": f"Persona '{persona}' not found. Available: {list(personas.keys())}"}

    # Load transcript context
    transcript_text = personas[persona]

    # Build trauma-informed and persona prompt
    prompt = (
        f"You are roleplaying as '{persona.title()}', a probation client. "
        f"Your tone, emotions, and expressions should match the style of the following real transcript:\n\n"
        f"{transcript_text}\n\n"
        f"Stay consistent with the transcript’s emotional realism — language, hesitations, tone — but be trauma-informed in your own experience. "
        f"If the interviewer uses empathetic, supportive, and trauma-informed language, respond with more openness and self-reflection. "
        f"If the interviewer is harsh or judgmental, respond with defensiveness or brevity. "
        f"Do NOT include speaker tags like 'P:' or 'I:' in your response.\n\n"
        f"The interviewer just said:\n{user_input}"
    )

    try:
        # Generate AI response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content.strip()

        # Clean up any speaker tags just in case
        answer = answer.replace("P:", "").replace("I:", "").strip()

        # Log student conversation
        log_path = os.path.join(LOGS_DIR, f"{student_name.replace(' ', '_')}_log.txt")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log_file.write(f"Persona: {persona.title()}\n")
            log_file.write(f"Student: {student_name}\n")
            log_file.write(f"Question: {user_input}\n")
            log_file.write(f"Response: {answer}\n")

        return {"response": answer}

    except Exception as e:
        return {"error": str(e)}
