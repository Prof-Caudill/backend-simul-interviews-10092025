import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Initialize OpenAI client with environment variable
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

from flask import Flask, jsonify, request
from openai import OpenAI
import os

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route('/')
def home():
    return jsonify({"message": "Backend is running!"})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_input}]
    )
    answer = response.choices[0].message.content
    return jsonify({"response": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)