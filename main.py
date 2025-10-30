import os
import json
import re
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sim-interviews-backend")

# ---------------------------
# FastAPI App + CORS
# ---------------------------
app = FastAPI(title="Simulated Interview Backend")

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://frontend-simul-interviews.onrender.com")

ALLOW_ORIGINS = [
    FRONTEND_URL,
    "https://frontend-simul-interviews-10092025.onrender.com",
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# OpenAI Setup
# ---------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------
# Logging Directory
# ---------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "interaction_logs.json")

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

LOG_DOWNLOAD_PASSWORD = os.getenv("LOG_DOWNLOAD_PASSWORD", "defaultpassword")

# ---------------------------
# Persona Definitions
# ---------------------------
PERSONAS = {
    "Maggie": {
        "age": 32,
        "gender": "F",
        "offense": "Possession & distribution of controlled substances; domestic violence",
        "risk": "Moderate-High",
        "style": "Anxious, cooperative, easily discouraged",
        "background": (
            "Maggie is a 32-year-old woman with repeated substance use issues and a history of domestic conflict. "
            "She wants to change but is frequently uncertain and anxious. She tends to hesitate, repeat herself, "
            "and ask for reassurance. She may ramble at times and use filler words ('um', 'you know', 'I dunno')."
        ),
    },
    "Simon": {
        "age": 47,
        "gender": "M",
        "offense": "Cultivation of marijuana; vehicle theft; probation violations; child support issues",
        "risk": "Moderate",
        "style": "Humble, cooperative, blue-collar, simple",
        "background": (
            "Simon is a pragmatic 47-year-old blue-collar man who downplays his problems. He speaks plainly, "
            "uses short sentences, and values family and work. He may be embarrassed and use casual phrasing."
        ),
    },
    "Rosa": {
        "age": 30,
        "gender": "F",
        "offense": "Possession of a controlled substance; probation violations",
        "risk": "Low-Medium",
        "style": "Anxious people-pleaser, trauma history",
        "background": (
            "Rosa is 30, polite but anxious, often apologetic and eager to please authority figures. "
            "She has trauma history and sometimes struggles to find words; she speaks hesitantly."
        ),
    },
    "Joseph": {
        "age": 37,
        "gender": "M",
        "offense": "Felony destruction of property; possession; probation violations",
        "risk": "Moderate-High",
        "style": "Reserved, defensive, unsure",
        "background": (
            "Joseph is 37 and guarded; he answers briefly and can be defensive. He may justify or deflect early in interviews "
            "and opens up slowly when treated respectfully."
        ),
    },
}

# ---------------------------
# Helpers
# ---------------------------
def build_system_prompt(persona_name: str) -> str:
    p = PERSONAS.get(persona_name)
    if not p:
        return ""
    return (
        f"You are roleplaying as {persona_name}, a probation client. Use the following background to stay in character:\n\n"
        f"{p['background']}\n\n"
        "Guidelines:\n"
        "- Stay fully in character as the named client. Do not reveal you are an AI or that this is a simulation.\n"
        "- Speak naturally and imperfectly: include hesitations, filler words, short sentences, occasional repetition.\n"
        "- Do NOT use academic, clinical, or criminological jargon.\n"
        "- Do NOT include the speaker labels 'P:' or 'I:' or any transcript-like markers.\n"
        "- Keep replies concise (2–6 sentences) unless the user asks follow-ups.\n"
        "- If unsure or uncomfortable, respond briefly and humanly, not analytically.\n"
    )

def clean_response_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'(^|\n)\s*[PI]\s*:\s*', r'\1', text)
    text = re.sub(r'\[?\b[PpIi]\b\]?\s*:\s*', '', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
    text = re.sub(r'(P:|I:)\s*$', '', text).strip()
    return text

def append_log(entry: dict):
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = []
    data.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ---------------------------
# Pydantic Models
# ---------------------------
class ChatPayload(BaseModel):
    message: str
    persona: str
    student_name: str = "unknown"

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
async def root():
    return {"message": "Backend running", "available_personas": list(PERSONAS.keys())}

@app.get("/personas")
async def list_personas():
    out = [
        {
            "name": name,
            "age": d["age"],
            "gender": d["gender"],
            "offense": d["offense"],
            "risk": d["risk"],
            "style": d["style"],
        }
        for name, d in PERSONAS.items()
    ]
    return JSONResponse(content=out)

@app.post("/chat")
async def chat(payload: ChatPayload):
    if payload.persona not in PERSONAS:
        raise HTTPException(status_code=404, detail=f"Persona '{payload.persona}' not found.")

    system_prompt = build_system_prompt(payload.persona)
    user_message = payload.message.strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.9,
            max_tokens=300,
        )

        raw_text = completion.choices[0].message.content if completion.choices else ""
        cleaned = clean_response_text(raw_text or "")

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "student_name": payload.student_name.strip() or "unknown",
            "persona": payload.persona,
            "message": user_message,
            "response": cleaned,
        }
        append_log(entry)

        return {"response": cleaned}

    except Exception as e:
        logger.exception("OpenAI request failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_logs")
async def download_logs(password: str):
    if password != LOG_DOWNLOAD_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid password")

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except FileNotFoundError:
        logs = []

    grouped = {}
    for entry in logs:
        student = entry.get("student_name", "unknown")
        grouped.setdefault(student, []).append(entry)

    grouped_file = os.path.join(LOG_DIR, "grouped_interaction_logs.json")
    with open(grouped_file, "w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2)

    return FileResponse(grouped_file, media_type="application/json", filename="grouped_interaction_logs.json")

@app.get("/health")
async def health():
    return {"status": "ok"}
