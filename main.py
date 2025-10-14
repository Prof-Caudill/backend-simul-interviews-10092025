from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
import os
import csv
import datetime

# === FastAPI setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Initialize OpenAI ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Logging setup ===
LOGS_DIR = "student_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# === Instructor access password ===
INSTRUCTOR_PASSWORD = os.getenv("INSTRUCTOR_PASSWORD", "teachsecure123")

# === Load persona transcripts ===
TRANSCRIPTS_DIR = "transcripts"
personas = {}
if os.path.exists(TRANSCRIPTS_DIR):
    for filename in os.listdir(TRANSCRIPTS_DIR):
        if filename.endswith(".txt"):
            persona_name = filename.replace(".txt", "").lower()
            with open(os.path.join(TRANSCRIPTS_DIR, filename), "r", encoding="utf-8") as f:
                personas[persona_name] = f.read()

# === Routes ===
@app.get("/")
def home():
    """Check backend status & list available personas."""
    return {"message": "Backend running!", "available_personas": list(personas.keys())}


@app.post("/chat")
async def chat(request: Request):
    """Handle chat messages between student and persona."""
    data = await request.json()
    user_input = data.get("message", "")
    persona = data.get("persona", "").lower()
    student_name = data.get("student_name", "anonymous")

    if not user_input or not persona:
        raise HTTPException(status_code=400, detail="Missing message or persona.")
    if persona not in personas:
        raise HTTPException(status_code=404, detail=f"Persona '{persona}' not found.")

    transcript_text = personas[persona]

    # --- Improved realism prompt ---
    system_prompt = (
        f"You are roleplaying as '{persona.title()}', a real probation client in an interview. "
        f"Use natural speech patterns: pauses, hesitations, slang, and emotional realism. "
        f"You may sound defensive, uncertain, or reflective. "
        f"Avoid sounding robotic or overly formal. "
        f"Base your tone, personality, and content on this real interview transcript:\n\n"
        f"{transcript_text}\n\n"
        f"Continue the interview as {persona.title()} — stay fully in character and never reveal you are AI.\n"
        f"The interviewer just asked:\n{user_input}"
    )

    try:
        # --- Generate AI response ---
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": system_prompt}],
        )
        raw_answer = response.choices[0].message.content.strip()

        # --- Clean up AI response ---
        cleaned_response = raw_answer.replace("P:", "").replace("I:", "").strip()

        # --- Log interaction by student name ---
        log_file = os.path.join(LOGS_DIR, f"{student_name.replace(' ', '_')}_log.csv")
        with open(log_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["timestamp", "student_name", "persona", "student_message", "ai_response"])
            writer.writerow([
                datetime.datetime.now().isoformat(),
                student_name,
                persona,
                user_input,
                cleaned_response
            ])

        return {"response": cleaned_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/instructor/logs")
async def get_logs(password: str):
    """Instructor-only access to download logs as a ZIP file."""
    if password != INSTRUCTOR_PASSWORD:
        raise HTTPException(status_code=403, detail="Invalid password.")

    import shutil
    zip_path = "all_student_logs.zip"
    shutil.make_archive("all_student_logs", "zip", LOGS_DIR)
    return FileResponse(zip_path, filename="student_logs.zip", media_type="application/zip")
