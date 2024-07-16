# app.py
from dotenv import load_dotenv
import os

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import appwrite
from appwrite.client import Client
from appwrite.services.database import Database
import googlemaps

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Initialize Appwrite client here Namann

database = Database(client)

# Initialize Google Maps client
gmaps = googlemaps.Client(key=os.getenv(''))

class Preferences(BaseModel):
    user_id: str
    preferences: str
    budget: int
    location: str

@app.post("/submit-preferences")
async def submit_preferences(preferences: Preferences):
    # Store user preferences in the database
    database.create_document(
        collection_id='users',
        document_id=preferences.user_id,
        data={
            'preferences': preferences.preferences,
            'budget': preferences.budget,
            'location': preferences.location
        }
    )
    return {"message": "Preferences submitted successfully"}

@app.get("/recommendations")
async def get_recommendations(user_id: str):
    # Fetch user data from the database
    user_data = database.get_document(collection_id='users', document_id=user_id)
    preferences = user_data['preferences']
    budget = user_data['budget']
    location = user_data['location']
    
    # Use Google Maps API to find places based on preferences and location
    places = gmaps.places_nearby(location=location, keyword=preferences, radius=5000)
    
    # Filter places based on budget (assuming places have a 'price_level' attribute)
    filtered_places = [place for place in places['results'] if place.get('price_level', 0) <= budget]
    
    # Store recommendations in the database
    database.create_document(
        collection_id='recommendations',
        document_id=user_id,
        data={'places': filtered_places}
    )
    
    return {"recommendations": filtered_places}