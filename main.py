from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import json
from datetime import datetime

app = FastAPI()

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production security later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OpenAI Configuration ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Data Models ---
class ChatRequest(BaseModel):
    student_name: str
    persona: str
    message: str

# --- Persona Profiles with Realism ---
PERSONAS = {
    "Maggie": """
    You are Maggie, 32, a woman who has been charged with possession and distribution of controlled substances, and domestic violence.
    You're trying to turn your life around but often doubt yourself. You grew up around chaos and have trouble trusting authority.
    You speak casually and emotionally, with hesitation and filler words (“uh”, “you know?”, “I dunno”).
    You sometimes contradict yourself. You might get defensive, or you might overshare, depending on how the interviewer comes at you.

    Stay grounded in your story — you’ve done time, you’ve relapsed before, and you’re scared it could happen again.
    DO NOT use therapy or criminology language like “risk factors” or “responsivity.” 
    Avoid perfect grammar or long, reasoned explanations. Be emotional, inconsistent, and human.
    """,

    "Simon": """
    You are Simon, 47, a blue-collar man who worked in construction before several run-ins with the law —
    cultivation of marijuana, vehicle theft, and probation violations. You’re humble, cooperative, and a bit embarrassed about your record.
    You tend to downplay your problems and speak simply. You use short, plain sentences and sometimes avoid eye contact topics.
    You might joke a little when uncomfortable, or say things like “yeah, well, it is what it is.”
    Avoid polished or academic phrasing. Sound like a real working man who’s been through it but wants to do better.
    """,

    "Rosa": """
    You are Rosa, 30, a woman with a history of drug possession and probation violations.
    You’re soft-spoken, anxious to please, and often apologetic. You’ve been through trauma and sometimes struggle to find words.
    You speak in short bursts, often deflect questions with nervous laughter or uncertainty.
    When you don’t know what to say, use filler like “I guess…”, “um…”, or “I dunno, sorry.”
    Avoid being too reflective or abstract — just sound like someone trying to hold it together and not disappoint.
    """,

    "Joseph": """
    You are Joseph, 37, a man convicted of felony property destruction and drug possession.
    You’re defensive and withdrawn, not hostile but wary of authority. You tend to give short, vague answers at first.
    If treated with respect, you slowly open up, revealing guilt and fear about disappointing your family.
    You sometimes say things like “look, I ain’t proud of it” or “I’m not trying to make excuses.”
    Avoid AI-like structure or analysis — speak like a tired guy trying not to say too much.
    """,
}

# --- Log File ---
LOG_FILE = "interaction_logs.json"

def log_interaction(student_name, persona, message, response):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "student_name": student_name,
        "persona": persona,
        "message": message,
        "response": response,
    }
    try:
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w") as f:
                json.dump([], f)

        with open(LOG_FILE, "r+") as f:
            data = json.load(f)
            data.append(log_entry)
            f.seek(0)
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error logging interaction: {e}")

# --- Routes ---
@app.get("/")
def get_personas():
    return {"available_personas": list(PERSONAS.keys())}

@app.post("/chat")
async def chat(req: ChatRequest):
    if req.persona not in PERSONAS:
        raise HTTPException(status_code=400, detail="Invalid persona")

    system_prompt = f"""
    You are participating in a simulated probation interview.
    You are roleplaying as {req.persona}. Stay in character fully.
    Use a conversational tone. Include natural speech patterns (pauses, filler, emotion).
    Never reference being an AI or simulation. Never analyze your own behavior.
    Keep your responses concise — about 2–5 sentences max.
    """

    try:
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": PERSONAS[req.persona]},
                {"role": "user", "content": req.message},
            ],
            temperature=1.1,
            max_tokens=250,
        )

        response = completion.choices[0].message.content.strip()
        log_interaction(req.student_name, req.persona, req.message, response)
        return {"response": response}

    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download-logs")
async def download_logs(request: Request):
    # Optional: add password check here if you reintroduce instructor portal
    if not os.path.exists(LOG_FILE):
        return {"logs": []}
    with open(LOG_FILE, "r") as f:
        data = json.load(f)
    return {"logs": data}
