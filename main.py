from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List



# for AI
from openai import OpenAI
import json

client = OpenAI(api_key="sk-ed86012e213a4b619178c9396f18f672", base_url="https://api.deepseek.com")

app = FastAPI()


# Set up CORS as before
origins = [
    "chrome-extension://dbgfjmgknleginhadcdfcjlfdhikjehn",
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
        # print(f"Received Twitter data: {item}")
        if item not in twitter_temp:
            twitter_temp.append(item)
            print("Twitter data added:", item)
            main_temp_database.append({"type": "twitter", "content": item})
        else:
            pass
            # print("Duplicate Twitter data ignored:", item)
            
    return {"message": "Twitter data received successfully", "count": len(payload)}


@app.post("/api/forex")
async def receive_forex_data(payload: ForexPayload):
    for item in payload.data:
        # print(f"Received Forex data: {item.title}")
        # Check for duplicates based on the title (or any unique property)
        if not any(existing["title"] == item.title for existing in forex_temp):
            forex_temp.append(item.dict())
            main_temp_database.append({"type": "forex", "content": item.dict()})

        else:
            pass
            # print("Duplicate Forex data ignored:", item.title)
    return {"message": "Forex data received successfully", "count": len(payload.data)}

# endpoint para sa extension (Ai)
@app.get("/api/get/deep_seek_data")
async def get_prediction():
    news_display = "Current Market Inputs:\n\n"
    for item in main_temp_database:
        if item['type'] == 'forex':
            impact = "🔥 HIGH IMPACT" if item['content']['isHighImpact'] else "⚠️ MEDIUM IMPACT"
            news_display += f"[FOREX] {impact} - {item['content']['title']} ({item['content']['time']})\n"
        elif item['type'] == 'twitter':
            news_display += f"[MARKET UPDATE] 🚨 {item['content']}\n"

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
                Bias: Direction don't include the word USD just the bias bullish or bearish
                Duration: Validity Duration
                Why: 1-2 sentences
                News that acts as driving force: indicate news that acts as driving force for the bias 
                
                """
            }
        ],
        temperature=0.4,  # Keep analytical
    ) 

    response_deep_seek = response.choices[0].message.content
    result = parse_trading_response(response_deep_seek)
    json_output = json.dumps(result, indent=2)

    print(f"AI Prediction: {json_output}")
    return {"data": json_output}



@app.get("/api/get/data")
async def get_main_temp_database():
    return {"data": main_temp_database}

# This shit run this API
# uvicorn main:app --reload

