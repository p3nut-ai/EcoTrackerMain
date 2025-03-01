import os
import json
from datetime import date
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# for AI (assuming you're using OpenAI's client with a custom base_url)
from openai import OpenAI

# Supabase client
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DEEPSEEK_API = os.getenv("DEEPSEEK_API")
CHROME_EXTENSION_ID = os.getenv("CHROME_EXTENSION_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


client = OpenAI(api_key=DEEPSEEK_API, base_url="https://api.deepseek.com")

app = FastAPI()

# Set up CORS as before
origins = [
    CHROME_EXTENSION_ID,
    "http://127.0.0.1:8000",
    "http://localhost:8000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForexItem(BaseModel):
    title: str
    time: str
    isHighImpact: bool

class ForexPayload(BaseModel):
    data: List[ForexItem]

# In-memory temporary database for news
twitter_temp = []  
forex_temp = []    
main_temp_database = twitter_temp + forex_temp

def parse_trading_response(text):
    parsed = {}
    lines = text.split('\n')
    for line in lines:
        if line.startswith("Bias:"):
            parsed["bias"] = line.split(": ")[1].strip()
        elif line.startswith("Duration:"):
            parsed["duration"] = line.split(": ")[1].strip()
        elif line.startswith("Why:"):
            parsed["rationale"] = line.split(": ")[1].strip()
        elif line.startswith("News that"):
            drivers = line.split(": ")[1].strip()
            parsed["news_drivers"] = [d.strip() for d in drivers.split(" and ")]
    return parsed

@app.post("/api/twitter")
async def receive_twitter_data(payload: List[str]):
    for item in payload:
        if item not in twitter_temp:
            twitter_temp.append(item)
            print("Twitter data added:", item)
            main_temp_database.append({"type": "twitter", "content": item})
        else:
            pass
    return {"message": "Twitter data received successfully", "count": len(payload)}

@app.post("/api/forex")
async def receive_forex_data(payload: ForexPayload):
    for item in payload.data:
        # Check for duplicates based on the title (or any unique property)
        if not any(existing["title"] == item.title for existing in forex_temp):
            forex_temp.append(item.dict())
            main_temp_database.append({"type": "forex", "content": item.dict()})
        else:
            pass
    return {"message": "Forex data received successfully", "count": len(payload.data)}

@app.get("/api/get/deep_seek_data")
async def get_prediction():
    today = date.today().isoformat()  
    
    # Check if an AI prediction for today's date already exists in Supabase
    pred_response = supabase.table("ai_predictions").select("*").eq("date", today).execute()
    
    if pred_response.data and len(pred_response.data) > 0:
        # Prediction record exists: fetch and return it
        stored_pred = pred_response.data[0]
        json_output = json.dumps({
            "bias": stored_pred.get("bias"),
            "duration": stored_pred.get("duration"),
            "rationale": stored_pred.get("rationale"),
            "news_drivers": stored_pred.get("news_drivers")
        }, indent=2)
        print(f"AI Prediction (from DB): {json_output}")
        return {"data": json_output}
    else:
        # No prediction for today: Build news display from main_temp_database
        news_display = "Current Market Inputs:\n\n"
        for item in main_temp_database:
            if item['type'] == 'forex':
                impact = "🔥 HIGH IMPACT" if item['content']['isHighImpact'] else "⚠️ MEDIUM IMPACT"
                news_display += f"[FOREX] {impact} - {item['content']['title']} ({item['content']['time']})\n"
            elif item['type'] == 'twitter':
                news_display += f"[MARKET UPDATE] 🚨 {item['content']}\n"
        
        # Call the AI API for prediction
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a USD-focused day trader. Analyze news/tweets and respond ONLY with: 1) Trading bias, 2) Validity duration, 3) 1-2 sentence rationale."
                },
                {
                    "role": "user",
                    "content": f"""Imagine you're a day trader that focuses on USD only. These are the current news - what Bias would you come up with this one?

{news_display}

Answer format:
    Bias: [Bullish/Bearish]
    Duration: Validity Duration (if 1-6 hours valid, use "1 Day")
    Why: 1-2 sentences explanation
    News that acts as driving force: indicate the news driving the bias
"""
                }
            ],
            temperature=1.0,
        )

        response_deep_seek = response.choices[0].message.content
        result = parse_trading_response(response_deep_seek)
        result['date'] = today  # add today's date to the record
        json_output = json.dumps(result, indent=2)
        print(f"AI Prediction: {json_output}")

        # Insert the prediction into the Supabase 'ai_predictions' table
        supabase.table("ai_predictions").insert(result).execute()
        
        # Optionally, also insert each news item into the Supabase 'news' table with today's date
        for item in main_temp_database:
            news_item = item.copy()
            news_item["date"] = today
            supabase.table("news").insert(news_item).execute()

        return {"data": json_output}

@app.get("/api/get/data")
async def get_main_temp_database():
    return {"data": main_temp_database}

# To run the API: uvicorn main:app --reload
