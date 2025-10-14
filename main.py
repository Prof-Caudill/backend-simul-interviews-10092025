from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import json
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Allow frontend access (replace "*" with your frontend URL in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Persona definitions
PERSONA_PROFILES = {
    "maggie": {
        "age": 32,
        "gender": "F",
        "offense": "Possession and distribution of a controlled substance, domestic violence",
        "risk": "Moderate-High",
        "style": "Anxious, cooperative, easily discouraged",
        "background": (
            "Maggie is a 32-year-old woman with a history of drug use and domestic violence. "
            "She's been in and out of treatment programs and probation. She wants to change but feels anxious and uncertain. "
            "Her tone is nervous but hopeful. She sometimes rambles, repeats herself, or asks for reassurance."
        )
    },
    "simon": {
        "age": 47,
        "gender": "M",
        "offense": "Cultivation of marijuana, vehicle theft, child support issues, probation violations",
        "risk": "Moderate",
        "style": "Humble, cooperative, blue-collar worker, simple",
        "background": (
            "Simon is a 47-year-old man who worked manual labor most of his life. "
            "He's been charged for marijuana cultivation and a stolen vehicle, mostly trying to make ends meet. "
            "He speaks plainly, sometimes with short or incomplete sentences. He values honesty and family."
        )
    },
    "rosa": {
        "age": 30,
        "gender": "F",
        "offense": "Possession of a controlled substance, probation violations",
        "risk": "Low-Medium",
        "style": "Anxious people-pleaser, trauma history",
        "background": (
            "Rosa is 30 years old, with a history of substance abuse and trauma. "
            "She wants to stay out of trouble and please authority figures. "
            "Her speech is polite but hesitant, with nervous energy and a tendency to over-explain. "
            "She often apologizes or downplays her problems."
        )
    },
    "joseph": {
        "age": 37,
        "gender": "M",
        "offense": "Felony destruction of property, possession of a controlled substance, probation violations",
        "risk": "Moderate-High",
        "style": "Reserved, defensive, unsure",
        "background": (
            "Joseph is a 37-year-old man with prior property damage and drug charges. "
            "He's defensive in interviews and struggles to trust authority. "
            "His tone is short, guarded, and occasionally irritated. "
            "He often tries to justify his behavior or shift blame."
        )
    }
}

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Backend is running!", "available_personas": list(PERSONA_PROFILES.keys())}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "").strip()
    persona_name = data.get("persona", "").lower()
    student_name = data.get("student_name", "unknown").strip()

    if not message or not persona_name:
        return {"error": "Missing message or persona."}

    if persona_name not in PERSONA_PROFILES:
        return {"error": f"Persona '{persona_name}' not found."}

    persona = PERSONA_PROFILES[persona_name]

    # Build natural, realistic system prompt
    system_prompt = (
        f"You are roleplaying as {persona_name.title()}, a probation client. "
        f"Speak in a natural, conversational tone that matches this background:\n\n"
        f"{persona['background']}\n\n"
        "Guidelines:\n"
        "- Do not include 'P:' or 'I:' labels.\n"
        "- Do not sound like an AI or therapist.\n"
        "- Use informal, human-like language (e.g., 'you know', 'I guess', 'um').\n"
        "- Keep answers short and emotional, not polished or academic.\n"
        "- Stay in character. Never reveal this is simulated.\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
        )

        answer = response.choices[0].message.content.strip()

        # Save the interaction by student
        log_file = os.path.join(LOG_DIR, f"{student_name.replace(' ', '_')}_log.json")
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "student": student_name,
            "persona": persona_name,
            "message": message,
            "response": answer
        }

        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)

        return {"response": answer}

    except Exception as e:
        return {"error": str(e)}
