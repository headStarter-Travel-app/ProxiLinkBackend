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

# Load environment variables
load_dotenv()

# Initialize Appwrite client
client = Client()
client.set_endpoint('https://cloud.appwrite.io/v1')
client.set_project('66930c61001b090ab206')
client.set_key(os.getenv('APPWRITE_API_KEY'))
client.set_self_signed()

# Appwrite configuration
appwrite_config = {
    "database_id": "66930e1000087eb0d4bd",
    "user_collection_id": "66930e5900107bc194dc",
    "preferences_collection_id": "6696016b00117bbf6352"
}

# Initialize FastAPI app
app = FastAPI(
    title="Restaurant Recommendation API",
    description="API for managing user preferences and generating restaurant recommendations",
    version="1.0.0",
)

# Global variable for Apple Maps token
# TODO: In production, initialize this as None

if (os.getenv('DEV')):
    global_maps_token = os.getenv('TOKEN_TEMP')
else:
    global_maps_token = None

# Function to update Apple Maps token


def update_apple_token():
    global global_maps_token
    global_maps_token = AppleAuth.generate_apple_token()
    print(f"Token updated at {datetime.now()}")


# Set up scheduler for token updates
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=update_apple_token,
    trigger=IntervalTrigger(hours=168),
    id='apple_token_update',
    name='Update Apple Token every 7 days',
    replace_existing=True
)
scheduler.start()

# Initialize Appwrite database service
database = Databases(client)

# Initialize Google Maps client
gmaps = googlemaps.Client(key=os.getenv('GOOGLE_MAPS_API'))

# Pydantic model for user preferences


class Preferences(BaseModel):
    # Temp
    user_id: str
    cuisine: List[str]
    entertainment: List[str]
<<<<<<< Updated upstream
    atmosphere: str
    social_interaction: str
    time_of_day: str
    spontaneity: str
=======
    socializing: SocialInteraction = SocialInteraction.BOTH
    Time: List[Time]
    shopping: Shopping = Shopping.SOMETIME
    family_friendly: bool
    learning: List[str]
    sports: List[str]
>>>>>>> Stashed changes


    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "cuisine": ["Italian", "Japanese"],
                "entertainment": ["Live Music", "Movies"],
                "atmosphere": "Casual",
                "social_interaction": "High",
                "time_of_day": "Evening",
                "spontaneity": "High"
            }
        }


@app.get("/get-apple-token", summary="Get Apple Maps Token")
async def get_apple_token():
    """
    Retrieve the current Apple Maps token. If not available, generate a new one.
    """
    global global_maps_token
    if global_maps_token is None:
        update_apple_token()
    return {"apple_token": global_maps_token}


@app.post("/submit-preferences", summary="Submit User Preferences")
async def submit_preferences(preferences: Preferences):
    """
    Submit and store user preferences in the database.
    """
    database.create_document(
        database_id=appwrite_config['database_id'],
        collection_id=appwrite_config['preferences_collection_id'],
        document_id=ID.unique(),
        data=preferences.model_dump()
    )
    return {"message": "Preferences submitted successfully"}


@app.get("/recommendations", summary="Get Restaurant Recommendations")
async def get_recommendations(user_id: str):
    """
    Generate restaurant recommendations based on user preferences and location.
    """
    # Fetch user data from the database
    user_data = database.get_document(
        database_id=appwrite_config['database_id'],
        collection_id=appwrite_config['user_collection_id'],
        document_id=user_id
    )
    preferences = user_data['preferences']
    location = user_data['location']

    # Use Google Maps API to find places based on preferences and location
    places = gmaps.places_nearby(
        location=location, keyword=preferences, radius=5000)

    # Store recommendations in the database
    database.create_document(
        database_id=appwrite_config['database_id'],
        collection_id='recommendations',
        document_id=user_id,
        data={'places': places['results']}
    )

    return {"recommendations": places['results']}


if (os.getenv('DEV')):
    if __name__ == "__main__":
        # For development use only
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    # Production use
    if __name__ == "__main__":
        update_apple_token()
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
