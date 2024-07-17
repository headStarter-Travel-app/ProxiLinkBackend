# app.py
from dotenv import load_dotenv
import os
from appleSetup import AppleAuth
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import appwrite
from appwrite.client import Client
from appwrite.services.users import Users
from appwrite.services.databases import Databases
from appwrite.id import ID
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime

import googlemaps


client = Client()

(client
 .set_endpoint('https://cloud.appwrite.io/v1')
 .set_project('66930c61001b090ab206')
 .set_key(os.getenv('APPWRITE_API_KEY'))  # Your secret API key
 .set_self_signed()
 )

appwrite_config = {
    "database_id": "66930e1000087eb0d4bd",
    "user_collection_id": "66930e5900107bc194dc",
    "preferences_collection_id": "6696016b00117bbf6352"
}

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# global_maps_token = None  #TODO IN PROD UNCOMMENT THIS
global_maps_token = os.getenv('TOKEN_TEMP')


def update_apple_token():
    global global_maps_token
    global_maps_token = AppleAuth.generate_apple_token()
    print(f"Token updated at {datetime.now()}")


scheduler = BackgroundScheduler()
scheduler.add_job(
    func=update_apple_token,
    trigger=IntervalTrigger(hours=168),
    id='apple_token_update',
    name='Update Apple Token every hour',
    replace_existing=True)
scheduler.start()

# Initialize Appwrite client here Namann
database = Databases(client)

# Initialize Google Maps client
gmaps = googlemaps.Client(key=os.getenv('GOOGLE_MAPS_API'))


class Preferences(BaseModel):
    user_id: str
    preferences: str  # Irrelevent remove
    budget: List[float]  # Irrelevent remove
    cuisine: List[str]


@app.get("/get-apple-token")
async def get_apple_token():
    global global_maps_token
    if global_maps_token is None:
        update_apple_token()
    return {"apple_token": global_maps_token}


@app.post("/submit-preferences")
async def submit_preferences(preferences: Preferences):
    # Store user preferences in the database
    database.create_document(
        database_id=appwrite_config['database_id'],

        collection_id=appwrite_config['preferences_collection_id'],
        document_id=ID.unique(),
        data={
            'users': preferences.user_id,
            "user_id": preferences.user_id,  # TODO: Delete one of these if the link works
            # 'preferences': preferences.preferences,
            # 'budget': preferences.budget,  #Rework, we dont want budget
            'cuisine': preferences.cuisine  # Not part of DB
        }
    )
    return {"message": "Preferences submitted successfully"}


@app.get("/recommendations")
async def get_recommendations(user_id: str):
    # Fetch user data from the database
    user_data = database.get_document(
        collection_id='users', document_id=user_id)
    preferences = user_data['preferences']
    budget = user_data['budget']
    location = user_data['location']

    # Use Google Maps API to find places based on preferences and location
    places = gmaps.places_nearby(
        location=location, keyword=preferences, radius=5000)

    # Filter places based on budget (assuming places have a 'price_level' attribute)
    filtered_places = [place for place in places['results']
                       if place.get('price_level', 0) <= budget]

    # Store recommendations in the database
    database.create_document(
        collection_id='recommendations',
        document_id=user_id,
        data={'places': filtered_places}
    )

    return {"recommendations": filtered_places}


# uvicorn app:app --reload
if __name__ == "__main__":
    # update_apple_token() # TODO IN PRODUncomment this line to update the token on startup ON DEV DONT DO IT USE ENV TEST TOKEN
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
