#!/usr/bin/env python
# coding: utf-8

# In[1]:


# app.py
from dotenv import load_dotenv
import os
from enum import Enum
from services.appleSetup import AppleAuth
from services.apple_maps import apple_maps_service
from services.google_maps import google_maps_service
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Any, Optional, Dict
import appwrite
from appwrite.client import Client
from appwrite.query import Query
from appwrite.services.users import Users
from appwrite.services.databases import Databases
from appwrite.id import ID
from apscheduler.schedulers.background import BackgroundScheduler
from featureFunctions import get_search_region
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import googlemaps
from appwrite.exception import AppwriteException
import httpx
import logging
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict

# Installed
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize Appwrite client
client = Client()
client.set_endpoint('https://cloud.appwrite.io/v1')
client.set_project('66930c61001b090ab206')
client.set_key(os.getenv('APPWRITE_API_KEY'))
client.set_self_signed()

appwrite_config = {
    "database_id": "66930e1000087eb0d4bd",
    "user_collection_id": "66930e5900107bc194dc",
    "preferences_collection_id": "6696016b00117bbf6352",
    "friends_collection_id": "friends",
    "locations_collection_id": "669d2a590010d4bf7d30",
    "groups_collection_id": "Groups",
    "contact_id": "66a419bb000fa8674a7e"
}

database = Databases(client)
users = Users(client)


# In[2]:


#1. We get all the preferences from each of the users that are passed in
#2: We get all the similar interests and the locations
#3: We query apple maps on the common interests and then 2-3 randomly selected ones, and then put the results in a array

def getPreferences(users: List[str]) -> Dict[str, Any]:
    res = defaultdict(int)
    for user in users:
        preferences = database.get_document(
            database_id=appwrite_config['database_id'],
            collection_id=appwrite_config['preferences_collection_id'],
            document_id=user
        )
        for key, value in preferences.items():
            if key == 'cuisine':
                for cuisine in value:
                    res[f"{cuisine} food"] += 1
            elif key == 'entertainment':
                for entertainment in value:
                    res[f"{entertainment}"] += 1
            elif key == 'shopping':
                if value == 'YES':
                    res['shopping'] += 1
            elif key == 'learning':
                for learning in value:
                    res[f"{learning}"] += 1
            elif key == 'sports':
                for sport in value:
                    res[f"{sport}"] += 1

    return res

# Example usage
preferences = getPreferences(["66996b7b0025b402922b", "66996d2e00314baa2a20", "669b45980030c00f3b8c", "669c735c001355ea24a7"])
print(preferences)


# In[3]:


class Location(BaseModel):
    lat: float
    lon: float

def calculate_centroid(locations: List[Location]) -> Dict[str, float]:
    """
    Calculate the centroid of given locations.
    """
    latitudes = [loc.lat for loc in locations]
    longitudes = [loc.lon for loc in locations]

    centroid_lat = sum(latitudes) / len(locations)
    centroid_lon = sum(longitudes) / len(locations)

    return {"lat": centroid_lat, "lon": centroid_lon}

#Get centroid with soem locations

locations = [Location(lat=38.98582939, lon= -76.937329584)]


# In[4]:


import random
def getTopInterests(preferences: Dict[str, int], top_n: int = 10) -> List[str]:
    # Filter preferences with values greater than 2
    filtered_preferences = {k: v for k, v in preferences.items() if v > 2}
    
    # Sort the filtered preferences by value in descending order
    sorted_preferences = sorted(filtered_preferences.items(), key=lambda item: item[1], reverse=True)
    
    # Get the top N keys
    top_interests = [k for k, v in sorted_preferences[:top_n]]
    
    # If there are fewer than N keys, randomly select additional keys from the remaining preferences
    if len(top_interests) < top_n:
        remaining_preferences = {k: v for k, v in preferences.items() if k not in top_interests}
        additional_interests = random.sample(list(remaining_preferences.keys()), top_n - len(top_interests))
        top_interests.extend(additional_interests)
    
    return top_interests

interestsPassin = getTopInterests(preferences)
print(interestsPassin)


# In[8]:


#1. pass in the preferenes and location, sort preferences by volume and get the 10 most popular (if rest are ones randomly choose)
#2. then make the proximitiy request and get the list
class ProximityRecommendationRequest(BaseModel):
    locations: List[Location]
    interests: List[str]


async def get_proximity_recommendations(request: ProximityRecommendationRequest):
    """
    Generate recommendations based on the centroid of provided user locations.
    """
    centroid = calculate_centroid(request.locations)
    print(centroid)
    try:
        all_recommendations = []
        for interest in request.interests:
            results = await apple_maps_service.search(interest, centroid['lat'], centroid['lon'])
            for result in results:
                if not result['category']:
                    result['category'] = interest  
                result['category2'] = interest

            all_recommendations.extend(results)

        # Sort recommendations by distance from centroid (if needed)
        sorted_recommendations = sorted(
            all_recommendations,
            key=lambda x: ((x['location']['lat'] - centroid['lat'])**2 +
                           (x['location']['lon'] - centroid['lon'])**2)**0.5
        )

        # Limit to top N recommendations if needed
        top_recommendations = sorted_recommendations

        return {"recommendations": sorted_recommendations}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting recommendations: {str(e)}")

locs = await get_proximity_recommendations(ProximityRecommendationRequest(locations=locations, interests=interestsPassin))
print(locs)


# In[9]:


import json
print(len(locs['recommendations']))


# In[10]:


print(json.dumps(locs['recommendations'], indent=4))


# In[ ]:




