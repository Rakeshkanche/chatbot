from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'chatbot_emotion_diagnosis_treatment.csv'))
df = pd.read_csv(DATASET_PATH).dropna()

class ChatRequest(BaseModel):
    history: list[str]
    mode: str  # "chat" or "design"

def detect_emotion(user_input):
    if 'AI-Detected Emotional State' in df.columns:
        for emotion in df['AI-Detected Emotional State'].dropna().unique():
            if emotion.lower() in user_input.lower():
                return emotion.lower()
    return None

def get_treatment_data(emotion):
    match = df[df['AI-Detected Emotional State'].str.lower() == emotion.lower()]
    if not match.empty:
        row = match.sample(1).iloc[0]
        return {
            "diagnosis": row['Diagnosis'],
            "medication": row['Medication'],
            "therapy": row['Therapy Type']
        }
    return None

@app.post("/chat/")
def chat(request: ChatRequest):
    user_input = request.history[-1]

    if request.mode == "chat":
        emotion = detect_emotion(user_input)
        if emotion:
            data = get_treatment_data(emotion)
            if data:
                return {
                    "answer": f"It seems you're feeling **{emotion}**. This may be linked to **{data['diagnosis']}**. "
                              f"Recommended therapy: **{data['therapy']}**. Medications: **{data['medication']}**."
                }

    # System Design or fallback to GPT
    messages = []
    for i, msg in enumerate(request.history[-6:]):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": msg})

    if request.mode == "design":
        messages.insert(0, {"role": "system", "content": "You are a helpful assistant that helps users design machine learning systems."})
    else:
        messages.insert(0, {"role": "system", "content": "You are a kind mental health support assistant."})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return {"answer": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"answer": "Sorry, something went wrong while processing your request."}
