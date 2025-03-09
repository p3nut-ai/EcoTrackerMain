import os
import json
import math
from datetime import date, timedelta, datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from termcolor import colored
import colorama

# For AI (using OpenAI's client with a custom base_url)
from openai import OpenAI

# Supabase client
from supabase import create_client, Client
from dotenv import load_dotenv

from apscheduler.schedulers.background import BackgroundScheduler

# Initialize colorama for Windows support.
colorama.init()

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

# Global in-memory caches (for temporary use only)
twitter_temp: List[str] = []      # Twitter data cache
forex_temp: List[dict] = []         # Forex data cache
main_temp_database: List[dict] = [] # Temporary news used for prediction
processed_dates: set = set()       # Processed prediction dates

# Global variable to hold the last known news count (if needed)
global_news_count = 0
new_prediction_flag = False
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
    Convert the predicted duration (e.g., "2 Days" or "1 Day") into a final expiration date.
    
    If the prediction is more than 1 day, subtract 1 (current day counts as day 1).
    If the prediction is 1 day, add 1 day to the current date.
    
    Return the resulting date in MM/DD/YYYY format.
    """
    try:
        predicted_days = int(predicted_duration.split()[0])
    except Exception as e:
        print(colored("Error parsing predicted duration: " + str(e), "red", "on_red"))
        predicted_days = 1

    if predicted_days > 1:
        effective_days = predicted_days - 1
    else:
        effective_days = predicted_days

    final_expiration = date.today() + timedelta(days=effective_days)
    print(colored(f"[+] Computed final expiration date: {final_expiration.strftime('%m/%d/%Y')} [+]", "white", "on_green"))
    return final_expiration.strftime("%m/%d/%Y")

def clear_temp_news():
    """Empty the temporary news cache (main_temp_database)."""
    main_temp_database.clear()
    print(colored("[+] Temporary news cache cleared. [+]", "white", "on_green"))

def generate_prediction_from_news(news_items: List[dict]) -> dict:
    """
    Build a news display string from the provided news_items,
    call the AI API to generate a prediction,
    convert the predicted duration into a final expiration date,
    and return a prediction dictionary.
    
    Also, store the compiled news display string in the "compiled_news" field.
    """
    news_display = "Current Market Inputs:\n\n"
    for item in news_items:
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
                    "content": ("You are a USD-focused day trader. Analyze news/tweets and respond ONLY with: "
                                "1) Trading bias, 2) Validity duration, 3) 1-2 sentence rationale.")
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
        print(colored("Error calling AI API: " + str(e), "red", "on_red"))
        raise HTTPException(status_code=500, detail="Error calling AI API")
    
    response_text = response.choices[0].message.content
    result = parse_trading_response(response_text)
    final_expiration_date = compute_final_expiration_date(result["duration"])
    result["duration"] = final_expiration_date
    result["date"] = date.today().isoformat()
    result["compiled_news"] = news_display
    print(colored("[+] Generated prediction from news successfully. [+]", "white", "on_green"))
    return result

def parse_trading_response(text: str) -> dict:
    """Parse the AI response text into a structured dictionary."""
    parsed = {}
    for line in text.split('\n'):
        if line.startswith("Bias:"):
            parsed["bias"] = line.split(": ", 1)[1].strip()
        elif line.startswith("Duration:"):
            parsed["duration"] = line.split(": ", 1)[1].strip()
        elif line.startswith("Why:"):
            parsed["rationale"] = line.split(": ", 1)[1].strip()
        elif line.startswith("News that"):
            drivers = line.split(": ", 1)[1].strip()
            parsed["news_drivers"] = [d.strip() for d in drivers.split(" and ")]
    print(colored("[+] Parsed AI response successfully. [+]", "white", "on_green"))
    return parsed

def update_prediction_if_new_news() -> None:
    """
    Check if there are new temporary news (in main_temp_database). If so, retrieve the previously
    compiled news from the stored prediction, combine it with the new temporary news, generate an
    updated prediction via the AI API, update the prediction record in the DB, and then clear the
    temporary news cache.
    """
    global global_news_count
    today = date.today().isoformat()
    if not main_temp_database:
        print(colored("[-] No new temporary news to update prediction. [-]", "yellow", "on_yellow"))
        return
    
    new_news_display = "New News:\n\n"
    for item in main_temp_database:
        if item.get("type") == "forex":
            impact = "🔥 HIGH IMPACT" if item.get("content", {}).get("isHighImpact") else "⚠️ MEDIUM IMPACT"
            new_news_display += f"[FOREX] {impact} - {item['content']['title']} ({item['content']['time']})\n"
        elif item.get("type") == "twitter":
            new_news_display += f"[MARKET UPDATE] 🚨 {item['content']}\n"
    
    pred_response = supabase.table("ai_predictions").select("*").eq("date", today).execute()
    if not pred_response.data:
        print(colored("[-] No existing prediction record found; generating new prediction. [-]", "yellow", "on_yellow"))
        updated_prediction = generate_prediction_from_news(main_temp_database)
    else:
        stored_pred = pred_response.data[0]
        old_compiled = stored_pred.get("compiled_news", "")
        combined_display = (old_compiled + "\n" + new_news_display) if old_compiled else new_news_display
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": ("You are a USD-focused day trader. Analyze news/tweets and respond ONLY with: "
                                    "1) Trading bias, 2) Validity duration, 3) 1-2 sentence rationale.")
                    },
                    {
                        "role": "user",
                        "content": f"""Imagine you're a day trader that focuses on USD only. These are the current news - what Bias would you come up with this one?

{combined_display}

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
            print(colored("Error calling AI API in update_prediction_if_new_news: " + str(e), "red", "on_red"))
            raise HTTPException(status_code=500, detail="Error calling AI API")
        response_text = response.choices[0].message.content
        updated_prediction = parse_trading_response(response_text)
        updated_prediction["duration"] = compute_final_expiration_date(updated_prediction["duration"])
        updated_prediction["date"] = today
        updated_prediction["compiled_news"] = combined_display
        print(colored("[+] Generated updated prediction successfully. [+]", "white", "on_green"))
    
    previous_count = stored_pred.get("news_count", 0) if pred_response.data else 0
    updated_prediction["news_count"] = previous_count + len(main_temp_database)
    supabase.table("ai_predictions").update(updated_prediction).eq("date", today).execute()
    clear_temp_news()
    global_news_count = 0
    print(colored("[+] Updated prediction record and cleared temporary news cache. [+]", "white", "on_green"))

def delete_expired_predictions() -> None:
    """
    Check all AI predictions in Supabase and delete those whose expiration date has passed
    (or are marked as "expired").
    """
    today = date.today()
    response = supabase.table("ai_predictions").select("*").execute()
    if not response.data:
        print(colored("[-] No predictions found in the table. [-]", "yellow", "on_yellow"))
        return

    for prediction in response.data:
        expiration_date_str = prediction.get("duration")
        if not expiration_date_str:
            print(colored("[-] Skipping prediction due to missing expiration date: " + str(prediction), "yellow", "on_yellow"))
            continue

        if expiration_date_str.lower() == "expired":
            supabase.table("ai_predictions").delete().eq("date", prediction.get("date")).execute()
            processed_dates.add(prediction.get("date"))
            print(colored("[!] Deleted expired prediction for date " + str(prediction.get("date")), "red", "on_red"))
            continue

        try:
            expiration_date = datetime.strptime(expiration_date_str, "%m/%d/%Y").date()
        except Exception as e:
            print(colored("[-] Invalid expiration date format: " + expiration_date_str + " Error: " + str(e), "red", "on_red"))
            continue

        if today > expiration_date:
            supabase.table("ai_predictions").delete().eq("date", prediction.get("date")).execute()
            processed_dates.add(prediction.get("date"))
            print(colored("[!] Deleted prediction with expiration date " + expiration_date_str, "red", "on_red"))
        else:
            print(colored("[+] Prediction with expiration date " + expiration_date_str + " is still valid.", "white", "on_green"))

# Scheduler to run deletion and update jobs periodically.
scheduler = BackgroundScheduler()
scheduler.add_job(delete_expired_predictions, "interval", minutes=1)
scheduler.add_job(update_prediction_if_new_news, "interval", minutes=1)

@app.on_event("startup")
def startup_event() -> None:
    scheduler.start()
    print(colored("[+] APScheduler started: deletion and update jobs scheduled every 1 minute. [+]", "white", "on_green"))

@app.on_event("shutdown")
def shutdown_event() -> None:
    scheduler.shutdown()
    print(colored("[!] APScheduler shutdown. [!]", "red", "on_red"))

@app.post("/api/twitter")
async def receive_twitter_data(payload: List[str]):
    """
    Receive Twitter data and insert it into the permanent "news" table if not already present.
    Duplicate checks are done by querying today's Twitter entries from the DB.
    """

    # issue here: cannot accept new tweets and send it to the DB
    current_date = date.today().isoformat()
    existing_response = supabase.table("news").select("content").eq("type", "twitter").eq("date", current_date).execute()
    existing_tweets = set()
    if existing_response.data:
        for item in existing_response.data:
            content = item.get("content")
            if content:
                existing_tweets.add(content.strip().lower())

    new_tweets = []
    for tweet in payload:
        normalized_tweet = tweet.strip().lower()
        if normalized_tweet not in existing_tweets:
            existing_tweets.add(normalized_tweet)
            twitter_temp.append(tweet)
            main_temp_database.append({"type": "twitter", "content": tweet})
            print(colored("[+] New Tweet data inserted. [+]", "white", "on_green"))
            new_tweets.append(tweet)
        else:
            print("[-] Duplicate Twitter data skipped: " + tweet)

    if new_tweets:
        insert_payload = [{"type": "twitter", "content": tweet, "date": current_date} for tweet in new_tweets]
        try:
            supabase.table("news").insert(insert_payload).execute()
        except Exception as e:
            print(colored("Error inserting new Twitter data: " + str(e), "red", "on_red"))

    return {"message": "Twitter data received successfully", "count": len(new_tweets)}

@app.post("/api/forex")
async def receive_forex_data(payload: ForexPayload):
    """
    Receive Forex data and insert it into the permanent "news" table if not already present.
    Duplicate checks are done by querying today's Forex entries (by title) from the DB.
    """
    current_date = date.today().isoformat()
    existing_response = supabase.table("news").select("content").eq("type", "forex").eq("date", current_date).execute()
    existing_titles = set()
    if existing_response.data:
        for item in existing_response.data:
            content = item.get("content")
            if content and isinstance(content, dict):
                title = content.get("title")
                if title:
                    existing_titles.add(title.strip().lower())
    new_forex = []
    for item in payload.data:
        if item.title.strip().lower() not in existing_titles:
            existing_titles.add(item.title.strip().lower())
            forex_temp.append(item.dict())
            main_temp_database.append({"type": "forex", "content": item.dict()})
            new_forex.append(item.dict())
            print(colored("[+] New Forex data inserted. [+]", "white", "on_green"))
        else:
            print(colored("[-] Duplicate Forex data skipped: " + item.title, "red", "on_red"))
    
    if new_forex:
        insert_payload = [{"type": "forex", "content": forex_item, "date": current_date} for forex_item in new_forex]
        try:
            supabase.table("news").insert(insert_payload).execute()
        except Exception as e:
            print(colored("Error inserting new Forex data: " + str(e), "red", "on_red"))
    
    return {"message": "Forex data received successfully", "count": len(new_forex)}


def set_new_prediction_flag(flag: bool):
    global new_prediction_flag
    new_prediction_flag = flag
    print(colored(f"[+] New prediction flag set to {flag} [+]", "white", "on_green" if flag else "on_yellow"))


@app.get("/api/new_prediction_status")
async def new_prediction_status():
    """
    Trigger the prediction endpoint internally to check for new predictions.
    Then return the current new_prediction_flag along with the prediction data.
    """
    # Call the get_prediction endpoint function internally.
    prediction_response = await get_prediction()
    # Extract the prediction data.
    prediction_data = prediction_response.get("data", None)
    # Return both the flag and the prediction.
    return {"new_prediction": new_prediction_flag, "prediction": prediction_data}




@app.get("/api/get/deep_seek_data")
async def get_prediction():
    """
    Get AI prediction for today.
    
    - If a prediction record for today exists and its expiration (duration) equals today's date,
      delete that prediction and all news rows (to welcome fresh data).
    - If a prediction exists and there is temporary news, update the prediction using the temporary news.
    - If no prediction exists, generate one from the temporary news.
    """
    global global_news_count
    try:
        today = date.today().isoformat()

        # Check if a prediction for today exists; if its duration equals today's date, delete it and all news.
        pred_response = supabase.table("ai_predictions").select("*").eq("date", today).execute()
        if pred_response.data and len(pred_response.data) > 0:
            stored_pred = pred_response.data[0]
            if stored_pred.get("duration") == today:
                print(colored("[-] Prediction expired (duration equals today). Deleting prediction and news rows. [-]", "red", "on_red"))
                supabase.table("ai_predictions").delete().eq("date", today).execute()
                supabase.table("news").delete().execute()
        
        # Query today's news from the permanent "news" table.
        news_response = supabase.table("news").select("*").eq("date", today).execute()
        current_news_items = news_response.data if news_response.data else []
        print(colored(f"[+] Total news in DB for today: {len(current_news_items)} [+]", "white", "on_green"))
        
        # Re-query prediction after possible deletion.
        pred_response = supabase.table("ai_predictions").select("*").eq("date", today).execute()
        
        # If a prediction exists:
        if pred_response.data and len(pred_response.data) > 0:
            stored_pred = pred_response.data[0]
            if main_temp_database:
                print(colored("[*] New temporary news detected. Updating prediction... [*]", "yellow", "on_yellow"))
                update_prediction_if_new_news()
                set_new_prediction_flag(True)
                pred_response = supabase.table("ai_predictions").select("*").eq("date", today).execute()
                stored_pred = pred_response.data[0]
                json_output = json.dumps(stored_pred, indent=2)
                return {"data": json_output}
            else:
                json_output = json.dumps(stored_pred, indent=2)
                return {"data": json_output}
        else:
            new_prediction = generate_prediction_from_news(main_temp_database)
            new_prediction["news_count"] = len(main_temp_database)
            supabase.table("ai_predictions").insert(new_prediction).execute()
            json_output = json.dumps(new_prediction, indent=2)
            clear_temp_news()
            global_news_count = 0
            set_new_prediction_flag(False)
            print(colored("[+] New prediction generated and temporary cache cleared. [+]", "white", "on_green"))
            return {"data": json_output}
    except Exception as e:
        print(colored("Error in get_prediction endpoint: " + str(e), "red", "on_red"))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get/data")
async def get_main_temp_database():
    """
    Return news items from the permanent "news" table for today in a simple list of dictionaries.
    
    Example output:
    [
      {'type': 'twitter', 'content': '*S&P 500 EXTENDS LOSSES DOWN 1.5%, ERASING ELECTION GAINS'},
      {'type': 'twitter', 'content': '*S&P 500 OPENS DOWN 0.8%, NASDAQ DOWN 1%'}
    ]
    """
    current_date = date.today().isoformat()
    news_response = supabase.table("news").select("type, content").eq("date", current_date).execute()
    news_items = news_response.data if news_response.data else []
    print(colored(f"[+] Returning {len(news_items)} news items from DB. [+]", "white", "on_green"))
    return {"data": news_items}

# To run the API: uvicorn main:app --reload
