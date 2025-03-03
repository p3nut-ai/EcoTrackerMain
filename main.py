import os
import json
import math
from datetime import date, timedelta, datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# For AI (using OpenAI's client with a custom base_url)
from openai import OpenAI

# Supabase client
from supabase import create_client, Client
from dotenv import load_dotenv

from apscheduler.schedulers.background import BackgroundScheduler

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DEEPSEEK_API = os.getenv("DEEPSEEK_API")
CHROME_EXTENSION_ID = os.getenv("CHROME_EXTENSION_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=DEEPSEEK_API, base_url="https://api.deepseek.com")

app = FastAPI()

# Set up CORS
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

# Pydantic models for Forex data
class ForexItem(BaseModel):
    title: str
    time: str
    isHighImpact: bool

class ForexPayload(BaseModel):
    data: List[ForexItem]

# Global in-memory temporary databases
twitter_temp: List[str] = []          # List of Twitter strings
forex_temp: List[dict] = []             # List of Forex items (as dicts)
main_temp_database: List[dict] = []     # Aggregated news items
processed_dates: set = set()           # Set of dates that have been processed

def parse_time_value(time_str: str) -> float:
    """
    Parse a time string (e.g., "47 hr ago", "2 days ago", "30 min ago")
    and return the value in hours.
    """
    parts = time_str.split()
    if len(parts) < 2:
        return 0
    try:
        value = float(parts[0])
    except ValueError:
        return 0
    unit = parts[1].lower()
    if "hr" in unit:
        return value
    elif "day" in unit:
        return value * 24
    elif "min" in unit:
        return math.ceil(value / 60)
    else:
        return 0

def compute_final_expiration_date(predicted_duration: str) -> str:
    """
    Compute the final expiration date using the following steps:
      1. Determine the starting point date:
         - Scan all forex news items in main_temp_database.
         - Convert each forex news item's time (e.g., "47 hr ago") into hours,
           determine the maximum, convert to days (round up), and subtract that many days from today.
      2. Extract the AI-predicted validity duration (e.g., "2 Days") as a number.
      3. Add the AI-predicted days to the starting point date.
      4. If the final expiration date equals today's date, return "expired",
         otherwise return the date formatted as MM/DD/YYYY.
    """
    # Step 1: Calculate maximum forex news age in hours.
    forex_hours = []
    for item in main_temp_database:
        if item.get("type") == "forex":
            time_str = item.get("content", {}).get("time", "")
            if time_str:
                hrs = parse_time_value(time_str)
                if hrs:
                    forex_hours.append(hrs)
    if forex_hours:
        max_hours = max(forex_hours)
        forex_days = int(math.ceil(max_hours / 24))
    else:
        forex_days = 0

    starting_point = date.today() - timedelta(days=forex_days)

    # Step 2: Extract predicted validity (in days) from the AI response.
    try:
        predicted_days = int(predicted_duration.split()[0])
    except Exception as e:
        print("Error parsing predicted duration:", e)
        predicted_days = 1

    # Step 3: Compute final expiration date.
    final_expiration = starting_point + timedelta(days=predicted_days)

    # Step 4: If final expiration equals today, mark as expired.
    if final_expiration == date.today():
        return "expired"
    else:
        return final_expiration.strftime("%m/%d/%Y")

def clean_news_table() -> None:
    """
    Delete all news items from the 'news' table that do not have today's date.
    """
    current_date = date.today().isoformat()
    print("Cleaning news table: deleting all news not from", current_date)
    supabase.table("news").delete().neq("date", current_date).execute()

def delete_expired_predictions() -> None:
    """
    Check all AI predictions in Supabase and delete those whose expiration date has passed
    (or are marked as "expired"). If at least one prediction is deleted, clean the news table.
    """
    today = date.today()
    print("Running delete_expired_predictions job for date:", today.strftime("%m/%d/%Y"))
    
    response = supabase.table("ai_predictions").select("*").execute()
    expired_found = False

    if not response.data:
        print("No predictions found in the table.")
    else:
        for prediction in response.data:
            expiration_date_str = prediction.get("duration")
            if not expiration_date_str:
                print("Skipping prediction due to missing expiration date:", prediction)
                continue

            # If the stored duration is "expired", delete immediately.
            if expiration_date_str.lower() == "expired":
                supabase.table("ai_predictions").delete().eq("date", prediction.get("date")).execute()
                processed_dates.add(prediction.get("date"))
                expired_found = True
                print("Deleted expired prediction for date", prediction.get("date"))
                continue

            try:
                expiration_date = datetime.strptime(expiration_date_str, "%m/%d/%Y").date()
            except Exception as e:
                print("Invalid expiration date format:", expiration_date_str, "Error:", e)
                continue

            print("Prediction expiration date:", expiration_date_str, "vs today:", today.strftime("%m/%d/%Y"))
            if today >= expiration_date:
                supabase.table("ai_predictions").delete().eq("date", prediction.get("date")).execute()
                processed_dates.add(prediction.get("date"))
                expired_found = True
                print("Deleted prediction with expiration date", expiration_date_str)
            else:
                print("Prediction with expiration date", expiration_date_str, "is still valid.")
    
    # If any prediction expired, clean the news table.
    if expired_found:
        clean_news_table()
        # Do NOT auto-generate a new prediction; instead, wait for user trigger.

# Initialize APScheduler to run the deletion job every minute.
scheduler = BackgroundScheduler()
scheduler.add_job(delete_expired_predictions, "interval", minutes=1)

@app.on_event("startup")
def startup_event() -> None:
    scheduler.start()
    print("APScheduler started: delete_expired_predictions job scheduled every 1 minute.")

@app.on_event("shutdown")
def shutdown_event() -> None:
    scheduler.shutdown()
    print("APScheduler shutdown.")

def parse_trading_response(text: str) -> dict:
    """Parse the AI response text into a structured dictionary."""
    parsed = {}
    lines = text.split('\n')
    for line in lines:
        if line.startswith("Bias:"):
            parsed["bias"] = line.split(": ", 1)[1].strip()
        elif line.startswith("Duration:"):
            parsed["duration"] = line.split(": ", 1)[1].strip()
        elif line.startswith("Why:"):
            parsed["rationale"] = line.split(": ", 1)[1].strip()
        elif line.startswith("News that"):
            drivers = line.split(": ", 1)[1].strip()
            parsed["news_drivers"] = [d.strip() for d in drivers.split(" and ")]
    return parsed

@app.post("/api/twitter")
async def receive_twitter_data(payload: List[str]):
    """Receive Twitter data and store it if not already present."""
    for item in payload:
        if item not in twitter_temp:
            twitter_temp.append(item)
            print("Twitter data added:", item)
            main_temp_database.append({"type": "twitter", "content": item})
        else:
            print("Duplicate Twitter data skipped:", item)
    return {"message": "Twitter data received successfully", "count": len(payload)}

@app.post("/api/forex")
async def receive_forex_data(payload: ForexPayload):
    """Receive Forex data and store it if not already present."""
    for item in payload.data:
        if not any(existing["title"] == item.title for existing in forex_temp):
            forex_temp.append(item.dict())
            main_temp_database.append({"type": "forex", "content": item.dict()})
            print("Forex data added:", item)
        else:
            print("Duplicate Forex data skipped:", item)
    return {"message": "Forex data received successfully", "count": len(payload.data)}

@app.get("/api/get/deep_seek_data")
async def get_prediction():
    """
    Get AI prediction for today or create a new one if not already processed.
    When triggered by the user, this function uses current-day news to generate a prediction.
    """
    print("Processed dates:", processed_dates)
    today = date.today().isoformat()

    # Check if prediction for today exists in Supabase.
    pred_response = supabase.table("ai_predictions").select("*").eq("date", today).execute()
    if pred_response.data and len(pred_response.data) > 0:
        stored_pred = pred_response.data[0]
        json_output = json.dumps({
            "bias": stored_pred.get("bias"),
            "duration": stored_pred.get("duration"),
            "rationale": stored_pred.get("rationale"),
            "news_drivers": stored_pred.get("news_drivers")
        }, indent=2)
        print("AI Prediction (from DB):", json_output)
        return {"data": json_output}
    else:
        # Prevent repeated input if prediction was processed before.
        if today in processed_dates:
            print(f"Prediction for date {today} was already processed and deleted. Not inserting new data.")
            return {"data": f"Prediction for date {today} was already processed."}

        # Build news display from main_temp_database.
        news_display = "Current Market Inputs:\n\n"
        for item in main_temp_database:
            if item.get("type") == "forex":
                impact = "🔥 HIGH IMPACT" if item.get("content", {}).get("isHighImpact") else "⚠️ MEDIUM IMPACT"
                news_display += f"[FOREX] {impact} - {item['content']['title']} ({item['content']['time']})\n"
            elif item.get("type") == "twitter":
                news_display += f"[MARKET UPDATE] 🚨 {item['content']}\n"

        try:
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
    Duration: Validity Duration (if 1-6 hours valid, use "1 Day" if it's more than that use "day" format )
    Why: 1-2 sentences explanation
    News that acts as driving force: indicate the news driving the bias
"""
                    }
                ],
                temperature=1.0,
            )
        except Exception as e:
            print("Error calling AI API:", e)
            raise HTTPException(status_code=500, detail="Error calling AI API")

        response_deep_seek = response.choices[0].message.content
        result = parse_trading_response(response_deep_seek)
        # Compute the final expiration date using the new logic.
        final_expiration_date = compute_final_expiration_date(result["duration"])
        result["duration"] = final_expiration_date
        result["date"] = today  # add today's date to the record

        json_output = json.dumps(result, indent=2)
        print("AI Prediction:", json_output)

        try:
            supabase.table("ai_predictions").insert(result).execute()
        except Exception as e:
            print("Error inserting prediction to Supabase:", e)
            raise HTTPException(status_code=500, detail="Error inserting prediction")

        # Upsert each news item into Supabase 'news' table with today's date.
        for item in main_temp_database:
            news_item = item.copy()
            news_item["date"] = today
            try:
                supabase.table("news").upsert(news_item).execute()
            except Exception as e:
                print("Error upserting news item to Supabase:", e)

        return {"data": json_output}

@app.get("/api/get/data")
async def get_main_temp_database():
    """Return the aggregated temporary news database."""
    return {"data": main_temp_database}

# To run the API: uvicorn main:app --reload
