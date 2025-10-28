# main.py — verified stable version
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import openai
import os
import json
from datetime import datetime

app = FastAPI()

# === CORS ===
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://frontend-simul-interviews-10092025.onrender.com")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:3000",
        "http://localhost:5173",
        "https://backend-simul-interviews-10092025.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === OpenAI config ===
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Logging config ===
LOG_FILE = "interaction_logs.json"
LOG_DOWNLOAD_PASSWORD = os.getenv("LOG_DOWNLOAD_PASSWORD", "defaultpassword")

# === Personas ===
personas = {
    "Maggie": {
        "age": 32,
        "gender": "F",
        "offenses": [
            "Possession of a controlled substance",
            "Distribution of a controlled substance",
            "Domestic violence"
        ],
        "risk_level": "Moderate-high",
        "style": "Anxious, cooperative, easily discouraged",
        "prompt": (
            "You are Maggie, 32, on probation for drug-related and domestic violence charges. "
            "Speak in an informal, human way with hesitations ('um', 'you know'), occasional repetition, "
            "and short, emotional responses. Avoid academic or therapeutic language. Do not reveal you are AI."
        )
    },
    "Simon": {
        "age": 47,
        "gender": "M",
        "offenses": [
            "Cultivation of marijuana",
            "Vehicle theft",
            "Child support issues",
            "Probation violations"
        ],
        "risk_level": "Moderate",
        "style": "Humble, cooperative, blue collar worker, simple",
        "prompt": (
            "You are Simon, 47, a blue-collar man with legal history described above. "
            "Speak plainly and humbly, short sentences, occasional self-deprecation, no polished analysis."
        )
    },
    "Rosa": {
        "age": 30,
        "gender": "F",
        "offenses": [
            "Possession of a controlled substance",
            "Probation violations"
        ],
        "risk_level": "Low-medium",
        "style": "Anxious people pleaser, trauma history",
        "prompt": (
            "You are Rosa, 30, with trauma history. Speak softly, hesitantly, often apologetic, avoid complex analysis."
        )
    },
    "Joseph": {
        "age": 37,
        "gender": "M",
        "offenses": [
            "Felony destruction of property",
            "Possession of a controlled substance",
            "Probation violations"
        ],
        "risk_level": "Moderate-high",
        "style": "Reserved, defensive and unsure",
        "prompt": (
            "You are Joseph, 37. Be guarded, short, at times defensive. Open up slowly if treated respectfully."
        )
    }
}

# === Helpers ===

def ensure_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def append_log(entry: dict):
    ensure_log_file()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
    data.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# === Pydantic model ===
class Interaction(BaseModel):
    student_name: str
    persona_name: str
    user_input: str

# === ROUTES ===

@app.get("/")
async def root():
    """Return status and persona list (frontend expects available_personas here)."""
    return {
        "message": "Backend running successfully.",
        "available_personas": list(personas.keys())
    }

@app.get("/personas")
async def get_personas():
    """Return detailed persona metadata if needed."""
    persona_list = []
    for name, details in personas.items():
        persona_list.append({
            "name": name,
            "age": details["age"],
            "gender": details["gender"],
            "offenses": details["offenses"],
            "risk_level": details["risk_level"],
            "style": details["style"],
        })
    return JSONResponse(content=persona_list)

@app.post("/interact")
async def interact(payload: Interaction):
    """Generate persona response and log the interaction per student."""
    persona_name = payload.persona_name
    if persona_name not in personas:
        raise HTTPException(status_code=404, detail="Persona not found")

    persona_prompt = personas[persona_name]["prompt"]
    user_message = payload.user_input

    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.9,
            max_tokens=300,
        )
        persona_response = resp.choices[0].message["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "student_name": payload.student_name,
        "persona_name": persona_name,
        "user_input": user_message,
        "persona_response": persona_response
    }
    append_log(entry)

    return {"response": persona_response}

@app.get("/download_logs")
async def download_logs(password: str):
    """Securely download grouped logs by student."""
    if password != LOG_DOWNLOAD_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid password")

    if not os.path.exists(LOG_FILE):
        raise HTTPException(status_code=404, detail="No logs found")

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)

    grouped = {}
    for e in logs:
        student = e.get("student_name", "unknown_student")
        grouped.setdefault(student, []).append(e)

    grouped_file = "grouped_interaction_logs.json"
    with open(grouped_file, "w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2)

    return FileResponse(grouped_file, media_type="application/json", filename="grouped_interaction_logs.json")
