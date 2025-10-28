# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import openai
import os

app = FastAPI()

# === CORS ===
FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "https://frontend-simul-interviews.onrender.com"  # replace with your frontend URL
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === OpenAI ===
openai.api_key = os.getenv("OPENAI_API_KEY")

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

# === ROUTES ===
@app.get("/")
async def root():
    """Return status and persona names (frontend expects available_personas)."""
    return {"message": "Backend running", "available_personas": list(personas.keys())}

@app.post("/chat")
async def chat(request: Request):
    """Generate persona response based on user input."""
    data = await request.json()
    persona_name = data.get("persona", "")
    user_input = data.get("message", "")

    if persona_name not in personas:
        return {"error": f"Persona '{persona_name}' not found."}

    persona_prompt = personas[persona_name]["prompt"]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.9,
            max_tokens=300
        )
        answer = response.choices[0].message.content
        return {"response": answer}
    except Exception as e:
        return {"error": str(e)}
