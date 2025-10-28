# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import json
from datetime import datetime
import openai

app = FastAPI()

# === CORS configuration ===
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://frontend-simul-interviews.onrender.com")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === OpenAI API Key ===
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Logging ===
LOG_FILE = "interaction_logs.json"
LOG_DOWNLOAD_PASSWORD = os.getenv("LOG_DOWNLOAD_PASSWORD", "defaultpassword")

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
            "Speak informally with hesitations ('um', 'you know'), short emotional responses. "
            "Do not reveal you are AI."
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
        "style": "Humble, cooperative, blue collar worker",
        "prompt": (
            "You are Simon, 47, a blue-collar man with legal history as above. "
            "Speak plainly, short sentences, occasional self-deprecation."
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
            "You are Rosa, 30, with trauma history. Speak softly, hesitantly, often apologetic, "
            "avoid complex analysis."
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

# === Pydantic model for interaction ===
class Interaction(BaseModel):
    student_name: str
    persona_name: str
    user_input: str

# === Routes ===
@app.get("/")
async def root():
    """Return backend status and available personas for frontend dropdown."""
    return {"message": "Backend running", "available_personas": list(personas.keys())}

@app.get("/personas")
async def get_personas():
    """Return detailed persona metadata."""
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
    """Generate persona response and log interaction per student."""
    if payload.persona_name not in personas:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    persona_prompt = personas[payload.persona_name]["prompt"]
    user_message = payload.user_input

    # OpenAI chat request
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

    # Log interaction
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "student_name": payload.student_name,
        "persona_name": payload.persona_name,
        "user_input": user_message,
        "persona_response": persona_response
    }
    append_log(entry)

    return {"response": persona_response}

@app.get("/download_logs")
async def download_logs(password: str):
    """Securely download logs grouped by student."""
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
