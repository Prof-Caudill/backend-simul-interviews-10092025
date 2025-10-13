import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# Initialize FastAPI app
app = FastAPI()

# Allow requests from all origins for testing (replace "*" with your frontend URL in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client (ensure OPENAI_API_KEY is set in Render)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set!")

client = OpenAI(api_key=OPENAI_KEY)

# Directory containing persona transcripts
TRANSCRIPTS_DIR = "transcripts"

# Load persona transcripts into memory
personas = {}
if os.path.exists(TRANSCRIPTS_DIR):
    for filename in os.listdir(TRANSCRIPTS_DIR):
        if filename.endswith(".txt"):
            persona_name = filename.replace(".txt", "").lower()
            with open(os.path.join(TRANSCRIPTS_DIR, filename), "r", encoding="utf-8") as f:
                personas[persona_name] = f.read()
else:
    os.makedirs(TRANSCRIPTS_DIR)  # Create folder if missing

# Health check and list available personas
@app.get("/")
def home():
    return {"message": "Backend is running!", "available_personas": list(personas.keys())}

# Chat endpoint
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    persona = data.get("persona", "").lower()

    if not persona or persona not in personas:
        return {"error": f"Persona '{persona}' not found. Available: {list(personas.keys())}"}

    transcript_text = personas[persona]
    prompt = (
        f"You are roleplaying as '{persona.title()}', a probation client. "
        f"Your tone, attitude, and word choice must stay consistent with the following real interview transcript:\n\n"
        f"{transcript_text}\n\n"
        f"Respond naturally to the interviewer. Do not reveal this is an AI simulation.\n\n"
        f"Interviewer asked:\n{user_input}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
        return {"response": answer}

    except Exception as e:
        return {"error": str(e)}
