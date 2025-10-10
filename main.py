import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")
client = OpenAI(api_key=api_key)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "Backend is running!"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        user_input = request.message

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_input}]
        )

        # Safely extract AI response
        answer = (
            response.choices[0].message.content
            if hasattr(response.choices[0].message, "content")
            else response.choices[0].message["content"]
        )

        return {"response": answer}

    except Exception as e:
        # This helps you debug errors in Render logs
        print(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
