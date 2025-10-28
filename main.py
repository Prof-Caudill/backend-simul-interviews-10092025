from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import openai
import os
import json
from datetime import datetime

# =========================================================
# CONFIGURATION
# =========================================================

app = FastAPI()

# Allow frontend access (adjust these if URLs change)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://frontend-simul-interviews-10092025.onrender.com",
        "http://localhost:5173",  # local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

LOG_FILE = "interaction_logs.json"
LOG_DOWNLOAD_PASSWORD = os.getenv("LOG_DOWNLOAD_PASSWORD", "defaultpassword")

# =========================================================
# DATA MODELS
# =========================================================

class Interaction(BaseModel):
    student_name: str
    persona_name: str
    user_input: str


# =========================================================
# PERSONA DATA
# =========================================================

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
            "You are Maggie, a 32-year-old woman on probation for drug-related and domestic violence charges. "
            "You’re anxious but want to do better. You try to be polite, open, and cooperative, but sometimes your anxiety or guilt shows. "
            "Speak naturally and realistically like someone with lived correctional experience, not like an AI."
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
            "You are Simon, a 47-year-old man on probation. You’re a humble, blue-collar guy trying to get back on your feet. "
            "You’ve made mistakes but you’re cooperative and realistic. Speak in a grounded, genuine way, with simple phrasing."
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
            "You are Rosa, a 30-year-old woman on probation for possession and probation violations. "
            "You’re anxious and tend to people-please. You have a trauma history but want to do well. "
            "Be realistic, occasionally hesitant, but earnest. Avoid sounding robotic or overly formal."
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
            "You are Joseph, a 37-year-old man on probation. You’ve struggled with substance use and anger management. "
            "You tend to be quiet, guarded, and defensive until you feel safe. Respond in a way that feels human and conflicted, "
            "sometimes short or uncertain, but genuine."
        )
    }
}


# =========================================================
# LOGGING
# =========================================================

def log_interaction(student_name, persona_name, user_input, persona_response):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "student_name": student_name,
        "persona_name": persona_name,
        "user_input": user_input,
        "persona_response": persona_response,
    }

    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
    except json.JSONDecodeError:
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
async def root():
    return {"message": "Simulated Interview Backend Active"}


@app.get("/personas")
async def get_personas():
    """
    Returns a list of persona names and their attributes
    for the dropdown menu in the frontend.
    """
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
async def interact(interaction: Interaction):
    persona = personas.get(interaction.persona_name)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    system_prompt = persona["prompt"]
    user_message = interaction.user_input

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.8
        )
        persona_response = response.choices[0].message["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Log interaction
    log_interaction(interaction.student_name, interaction.persona_name, user_message, persona_response)

    return {"response": persona_response}


@app.get("/download_logs")
async def download_logs(password: str):
    """
    Secure endpoint to download interaction logs grouped by student name.
    Example: /download_logs?password=yourpassword
    """
    if password != LOG_DOWNLOAD_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid password")

    if not os.path.exists(LOG_FILE):
        raise HTTPException(status_code=404, detail="No logs found")

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)

    grouped_logs = {}
    for entry in logs:
        student = entry.get("student_name", "unknown_student")
        grouped_logs.setdefault(student, []).append(entry)

    grouped_file = "grouped_interaction_logs.json"
    with open(grouped_file, "w", encoding="utf-8") as f:
        json.dump(grouped_logs, f, indent=2)

    return FileResponse(
        grouped_file,
        media_type="application/json",
        filename="grouped_interaction_logs.json"
    )
