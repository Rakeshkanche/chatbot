from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Load dataset
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'chatbot_emotion_diagnosis_treatment.csv'))

try:
    df = pd.read_csv(DATASET_PATH).dropna()
    print("✅ Dataset loaded. Columns:", df.columns.tolist())
except Exception as e:
    print("❌ Error loading dataset:", e)
    df = pd.DataFrame()  # fallback to empty DataFrame

# API request body
class ChatRequest(BaseModel):
    history: list[str]

# Keyword mapping for emotion detection
emotion_keywords = {
    "anxious": ["nervous", "worried", "tense", "panicky"],
    "sad": ["depressed", "unhappy", "down", "tearful"],
    "irritable": ["annoyed", "frustrated", "agitated"],
    "hopeless": ["empty", "lost", "no future", "defeated"],
    "lonely": ["isolated", "alone", "abandoned"]
}

# Emotion detection with keyword fallback
def detect_emotion(user_input):
    user_input = user_input.lower()

    if 'AI-Detected Emotional State' in df.columns:
        for emotion in df['AI-Detected Emotional State'].dropna().unique():
            if emotion.lower() in user_input:
                return emotion.lower()

    for emotion, keywords in emotion_keywords.items():
        for word in keywords:
            if word in user_input:
                return emotion
    return None

# Get diagnosis/therapy based on emotion
def get_diagnosis_treatment(emotion):
    try:
        match = df[df['AI-Detected Emotional State'].str.lower() == emotion.lower()]
        if not match.empty:
            row = match.sample(1).iloc[0]
            return {
                "diagnosis": row['Diagnosis'],
                "medication": row['Medication'],
                "therapy": row['Therapy Type']
            }
    except Exception as e:
        print(f"❌ Error matching emotion data: {e}")
    return None

@app.post("/chat/")
def chat(request: ChatRequest):
    user_input = request.history[-1]
    print(f"📥 User input received: {user_input}")

    emotion = detect_emotion(user_input)
    print(f"🧠 Detected emotion: {emotion}")

    if emotion:
        data = get_diagnosis_treatment(emotion)
        if data:
            return {
                "answer": f"It seems you're feeling **{emotion}**. This may be linked to **{data['diagnosis']}**. "
                          f"Recommended therapy: **{data['therapy']}**. Medications: **{data['medication']}**."
            }

    print("🤖 Falling back to GPT-4 response...")

    try:
        conversation = []
        for i, msg in enumerate(request.history[-6:]):
            role = "user" if i % 2 == 0 else "assistant"
            conversation.append({"role": role, "content": msg})

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a kind and supportive mental health chatbot."},
                *conversation
            ]
        )

        answer = response.choices[0].message.content.strip()
        return {"answer": answer}

    except Exception as e:
        print("❌ GPT fallback failed:", e)
        return {"answer": "Sorry, I'm having trouble replying right now."}

