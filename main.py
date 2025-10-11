from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os

# Initialize FastAPI app
app = FastAPI()

# CORS setup (replace * with your frontend domain later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load transcript for persona
TRANSCRIPT_PATH = "transcripts/Maggie.txt"

with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    transcript_text = f.read()

@app.get("/")
def home():
    return {"message": "Backend is running!"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")

    # Construct the system + user prompt
    prompt = (
        f"You are roleplaying as 'Maggie', a probation client. "
        f"Your tone, perspective, and responses should be consistent with the following transcript of Maggie's real interview:\n\n"
        f"{transcript_text}\n\n"
        f"Now, continue a realistic conversation as Maggie with a graduate student interviewer. "
        f"Only respond as Maggie, not as an AI. "
        f"The student's next question is:\n\n{user_input}\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        return {"response": answer}

    except Exception as e:
        return {"error": str(e)}
