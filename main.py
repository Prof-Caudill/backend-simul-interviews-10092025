import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI client with environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define a request model for the /chat endpoint
class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "Backend is running!"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """Handles chat messages from the frontend"""
    user_input = request.message

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_input}]
    )

    answer = response.choices[0].message.content
    return {"response": answer}
