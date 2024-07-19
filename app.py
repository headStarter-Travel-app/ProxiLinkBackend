# app.py
from dotenv import load_dotenv
import os
from enum import Enum
from appleSetup import AppleAuth
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Any, Optional
import appwrite
from appwrite.client import Client
from appwrite.query import Query
from appwrite.services.users import Users
from appwrite.services.databases import Databases
from appwrite.id import ID
from apscheduler.schedulers.background import BackgroundScheduler
from featureFunctions import calculate_centroid
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import googlemaps
import httpx

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
    title="Proxi Link AI API",
    description="API for Proxi Link App",
    version="1.0.0",
)

GOOGLE_API_KEY = os.getenv('GOOGLE_MAPS_API')

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

# model for location google maps api call (recommendations)
class Location(BaseModel):
    lat: float
    lon: float



class SocialInteraction(str, Enum):
    ENERGETIC = "ENERGETIC"
    RELAXED = "RELAXED"
    BOTH = "BOTH"


class Time(str, Enum):
    MORNING = "MORNING"
    AFTERNOON = "AFTERNOON"
    EVENING = "EVENING"
    NIGHT = "NIGHT"


class Shopping(str, Enum):
    YES = "YES"
    SOMETIME = "SOMETIME"
    NO = "NO"


# {
#   "user_id": "66943f0900123a1015ee",
#   "users": "66943f0900123a1015ee",
#   "cuisine": ["Italian", "Japanese", "Mexican"],
#   "atmosphere": ["Cozy", "Modern"],
#   "entertainment": ["Live Music", "Sports Screening"],
#   "socializing": "BOTH",
#   "Time": ["EVENING", "NIGHT"],
#   "shopping": "SOMETIME",
#   "family_friendly": true,
#   "learning": ["History", "Art"]
# }

# NOTE: MAKE SURE IN FRKONT END ENUSM ARE PROPERLY USED
class Preferences(BaseModel):
    # Temp
    user_id: str
    users: Any
    cuisine: List[str]
    atmosphere: List[str]
    entertainment: List[str]
    socializing: SocialInteraction = SocialInteraction.BOTH
    Time: List[Time]
    shopping: Shopping = Shopping.SOMETIME
    family_friendly: bool
    learning: List[str]
    sports: List[str]


@app.get("/", summary="Root")
async def root():
    return {"message": "Welcome to the Proxi Link API"}


@app.get("/get-apple-token", summary="Get Apple Maps Token")
async def get_apple_token():
    """
    Retrieve the current Apple Maps token. If not available, generate a new one.
    """
    global global_maps_token
    if global_maps_token is None:
        update_apple_token()
    return {"apple_token": global_maps_token}


class AccountInfo(BaseModel):
    uid: str
    firstName: str
    lastName: str
    address: Optional[str] = None


@app.post("/update-account", summary="Update Account")
async def update_account(account: AccountInfo):
    """
    Update user account in the database
    """
    try:
        account_dict = account.model_dump()
        existing_account = database.get_document(
            database_id=appwrite_config['database_id'],
            collection_id=appwrite_config['user_collection_id'],
            document_id=account_dict['uid']
        )
        if account_dict['address'] is None:
            account_dict['address'] = existing_account['address']

        result = database.update_document(
            database_id=appwrite_config['database_id'],
            collection_id=appwrite_config['user_collection_id'],
            document_id=account_dict['uid'],
            data=account_dict
        )

        return {"message": "User account updated successfully", "document_id": result['$id']}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating user account: {str(e)}")


@app.post("/submit-preferences", summary="Submit User Preferences")
async def submit_preferences(preferences: Preferences):
    """
    Submit and store user preferences in the database.
    """
    try:
        preferences_dict = preferences.model_dump()
        unique_id = preferences_dict['user_id']

        result = database.create_document(
            database_id=appwrite_config['database_id'],
            collection_id=appwrite_config['preferences_collection_id'],
            document_id=unique_id,
            data=preferences_dict
        )

        return {"message": "Preferences submitted successfully", "document_id": result['$id']}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error submitting preferences: {str(e)}")


@app.post("/update-preferences", summary="Update User Preferences")
async def update_preferences(preferences: Preferences):
    """
    Update and store user preferences in the database.
    """
    try:
        preferences_dict = preferences.model_dump()

        result = database.update_document(
            database_id=appwrite_config['database_id'],
            collection_id=appwrite_config['preferences_collection_id'],
            document_id=preferences_dict['user_id'],
            data=preferences_dict  # data is being sent as a dict json
        )
        return {"message": "Preferences updated successfully", "document_id": result['$id']}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating preferences: {str(e)}")


# Recommendation using preferences in user collection and location use later
@app.get("/recommendations", summary="Get Recommendations")
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
        document_id=ID.unique(),
        data={'places': places['results']}
    )

    return {"recommendations": places['results']}


# Default recommendations based on only location
@app.post("/get-recommendations", summary="Get Recommendations")
async def get_recommendations(location: Location):
    """
    Return the closest places from Google Places API based on location and category.
    """
    # Default recommendations based on only location
    dummy_recommendations = [
        {"type": "entertainment", "name": "Live Music Venue", "location": {"lat": 37.78825, "lon": -122.4324}},
        {"type": "food", "name": "Italian Restaurant", "location": {"lat": 37.78925, "lon": -122.4314}},
        {"type": "shopping", "name": "Local Bookstore", "location": {"lat": 37.79025, "lon": -122.4304}},
        {"type": "sightseeing", "name": "Golden Gate Park", "location": {"lat": 37.79125, "lon": -122.4294}},
        {"type": "cafe", "name": "Cozy Cafe", "location": {"lat": 37.79225, "lon": -122.4284}}
    ]
    
    # Uncomment and use this section when you want to use Google Places API
    # try:
    #     async with httpx.AsyncClient() as client:
    #         recommendations = []
    #         categories = ["music", "entertainment", "food", "museum", "park"]
    #         
    #         for category in categories:
    #             response = await client.get(
    #                 "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
    #                 params={
    #                     "location": f"{location.lat},{location.lon}",
    #                     "radius": 5000,  # Radius in meters
    #                     "type": category,
    #                     "key": GOOGLE_API_KEY
    #                 }
    #             )
    #             results = response.json().get("results", [])
    #             
    #             # Sort places by distance (by their location)
    #             sorted_places = sorted(
    #                 results,
    #                 key=lambda place: (
    #                     (place["geometry"]["location"]["lat"] - location.lat) ** 2 +
    #                     (place["geometry"]["location"]["lng"] - location.lon) ** 2
    #                 )
    #             )
    #             
    #             # Add top 5 closest places to recommendations
    #             for place in sorted_places[:5]:
    #                 recommendations.append({
    #                     "type": category,
    #                     "name": place.get("name"),
    #                     "location": {
    #                         "lat": place.get("geometry", {}).get("location", {}).get("lat"),
    #                         "lon": place.get("geometry", {}).get("location", {}).get("lng")
    #                     }
    #                 })
    #         
    #         return {"recommendations": recommendations}
    # except httpx.HTTPStatusError as e:
    #     raise HTTPException(status_code=e.response.status_code, detail=f"Google Maps API error: {str(e)}")
    # except httpx.RequestError as e:
    #     raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")

    return {"recommendations": dummy_recommendations}

@app.post("/get-proximity-recommendations", summary="Get Recommendations Based on Proximity")
async def get_proximity_recommendations(locations: List[Location]):
    """
    Generate recommendations based on the centroid of provided user locations.
    """
    # Calculate the centroid of the given locations
    centroid = calculate_centroid(locations)
    
    # Use Google Maps API to find places near the centroid
    try:
        response = await httpx.AsyncClient().get(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params={
                "location": f"{centroid['lat']},{centroid['lon']}",
                "radius": 5000,  # Radius in meters
                "key": GOOGLE_API_KEY
            }
        )
        results = response.json().get("results", [])
        
        # Format the recommendations
        recommendations = [
            {
                "name": place.get("name"),
                "location": {
                    "lat": place.get("geometry", {}).get("location", {}).get("lat"),
                    "lon": place.get("geometry", {}).get("location", {}).get("lng")
                }
            }
            for place in results
        ]
        
        return {"recommendations": recommendations}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Google Maps API error: {str(e)}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")


# uvicorn app:app --reload

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
