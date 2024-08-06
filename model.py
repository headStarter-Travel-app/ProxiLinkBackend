import asyncio
from dotenv import load_dotenv
import os
from enum import Enum
from services.appleSetup import AppleAuth
from services.apple_maps import apple_maps_service
from services.google_maps import google_maps_service
from pydantic import BaseModel
from typing import List, Any, Optional, Dict
import appwrite
from appwrite.client import Client
from appwrite.query import Query
from appwrite.services.users import Users
from appwrite.services.databases import Databases
from appwrite.id import ID
from datetime import datetime
import googlemaps
import logging
from collections import defaultdict
import random
import json
from pprint import pprint
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


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


class Location(BaseModel):
    lat: float
    lon: float


class ProximityRecommendationRequest(BaseModel):
    locations: List[Location]
    interests: List[str]


class AiModel:
    romantic_date = [
        "Cafe", "Restaurant", "Bakery", "AmusementPark", "Beach", "Winery",
        "Theater", "MovieTheater", "Park", "Zoo", "Aquarium", "Store",
        "MiniGolf", "Bowling", "MusicVenue", "Store", "Mall"
    ]

    family_outing = [
        "AmusementPark", "Zoo", "Aquarium", "Park", "Playground", "MovieTheater",
        "Museum", "NationalPark", "Beach", "Campground", "FoodMarket"
    ]

    outdoor_adventure = [
        "NationalPark", "Park", "Beach", "Hiking", "Kayaking", "Fishing",
        "Golf", "MiniGolf", "RockClimbing", "RVPark", "SkatePark", "Skating",
        "Skiing", "Surfing", "Swimming", "Tennis", "Volleyball"
    ]

    educational_trip = [
        "Museum", "Library", "Aquarium", "NationalPark", "Planetarium", "Zoo",
        "University", "Landmark", "NationalMonument", "ReligiousSite"
    ]

    night_out = [
        "Nightlife", "Brewery", "Restaurant", "MovieTheater", "Theater",
        "MusicVenue", "Casino", "Bar", "Store", "Winery"
    ]

    relaxation_and_wellness = [
        "Beach", "Spa", "FitnessCenter", "Park", "Yoga", "MeditationCenter"
    ]

    sports_and_fitness = [
        "Stadium", "FitnessCenter", "Golf", "Tennis", "Basketball", "Soccer",
        "Baseball", "Swimming", "Volleyball", "Bowling", "RockClimbing",
        "Hiking", "Kayaking", "Surfing", "Skating", "Skiing", "SkatePark"
    ]

    shopping_spree = [
        "Store", "FoodMarket", "Mall", "Pharmacy"
    ]

    kids_fun_day = [
        "AmusementPark", "Zoo", "Aquarium", "Park", "Playground", "MovieTheater",
        "MiniGolf", "Bowling", "Fairground", "GoKart"
    ]

    historical_and_cultural_exploration = [
        "Museum", "Castle", "Fortress", "Landmark", "NationalMonument",
        "ReligiousSite", "Planetarium", "Fairground", "ConventionCenter"
    ]

    Vacation = [
        "Hotel", "Beach", "NationalPark", "Park", "Winery", "Campground",
        "Marina", "Skiing", "RVPark", "Store"
    ]

    food_and_drink = [
        "Restaurant", "Cafe", "Bakery", "Brewery", "Winery", "FoodMarket"
    ]

    theme_categories = {
        "romantic_date": romantic_date,
        "family_outing": family_outing,
        "outdoor_adventure": outdoor_adventure,
        "educational_trip": educational_trip,
        "night_out": night_out,
        "relaxation_and_wellness": relaxation_and_wellness,
        "sports_and_fitness": sports_and_fitness,
        "shopping_spree": shopping_spree,
        "kids_fun_day": kids_fun_day,
        "historical_and_cultural_exploration": historical_and_cultural_exploration,
        "vacation": Vacation,
        "food_and_drink": food_and_drink
    }

    def __init__(self, users: List[str], location: List[Location], theme: str):
        self.users = users
        self.location = location
        self.preferences = None
        self.top_interests = None
        self.locationsList = None
        self.theme = self.__class__.theme_categories[theme]

    async def initialize(self):
        # 1. Get preferences
        self.preferences = self.getPreferences(self.users)

        # 2 & 3: Get Top interests
        self.top_interests = self.getTopInterests(self.preferences)

        # 4: Get recommendations
        requestRec = ProximityRecommendationRequest(
            locations=self.location, interests=self.top_interests)
        locs = await self.get_proximity_recommendations(requestRec)

        # 5: Store recommendations as json
        self.locationsList = (locs['recommendations'])

    @classmethod
    async def create(cls, users: List[str], location: List[Location]):
        instance = cls(users, location)
        await instance.initialize()
        return instance

    def getPreferences(self, users: List[str]) -> Dict[str, Any]:
        '''
        # Example usage
        preferences = getPreferences(["66996b7b0025b402922b", "66996d2e00314baa2a20", "669b45980030c00f3b8c", "669c735c001355ea24a7"])
        print(preferences)
        returns:
        defaultdict(<class 'int'>, {'Japanese food': 1, 'Italian food': 2, 'Arcade': 2, 'Parks': 2, 'shopping': 4, 'Culture': 2, 'Golf': 2, 'Indian food': 1, 'British food': 1, 'Eating': 1, 'Beach': 2, 'Museums': 1, 'Football': 1, 'Korean food': 1, 'Spas': 2, 'Go Kart': 2, 'Mexican food': 1, 'Belgian food': 1, 'Pizza food': 1, 'Cinemas': 1, 'Bars': 1, 'Music': 1, 'Theme Parks': 1, 'Nightlife': 1, 'Club': 1, 'Historical Sites': 1, 'Soccer': 1, 'Aquatic Sports': 1, 'Live Sports': 1})


        '''
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

    def calculate_centroid(self, locations: List[Location]) -> Dict[str, float]:
        """
        Calculate the centroid of given locations.
        returns: [Location(lat=38.98582939, lon=-76.937329584)]
        """
        latitudes = [loc.lat for loc in locations]
        longitudes = [loc.lon for loc in locations]

        centroid_lat = sum(latitudes) / len(locations)
        centroid_lon = sum(longitudes) / len(locations)

        return {"lat": centroid_lat, "lon": centroid_lon}

    def getTopInterests(self, preferences: Dict[str, int], top_n: int = 10) -> List[str]:
        '''
        Using the preferences from "get preferences" function, get the top N interests
        Example usage:
        interestsPassin = getTopInterests(preferences)
        returns: ['shopping', 'Go Kart', 'Arcade', 'Theme Parks', 'Aquatic Sports', 'Japanese food', 'Museums', 'Football', 'Culture', 'Music']
        '''
        # Filter preferences with values greater than 2
        filtered_preferences = {k: v for k, v in preferences.items() if v > 2}

        sorted_preferences = sorted(
            filtered_preferences.items(), key=lambda item: item[1], reverse=True)

        top_interests = [k for k, v in sorted_preferences[:top_n]]

        # If there are fewer than N keys, randomly select additional keys from the remaining preferences
        if len(top_interests) < top_n:
            remaining_preferences = {
                k: v for k, v in preferences.items() if k not in top_interests}
            additional_interests = random.sample(
                list(remaining_preferences.keys()), top_n - len(top_interests))
            top_interests.extend(additional_interests)

        return top_interests

    async def get_proximity_recommendations(self, request: ProximityRecommendationRequest):
        '''
        Get recommendations based on the user's preferences and location.
        Example usage:
        recommendations = get_proximity_recommendations(ProximityRecommendationRequest(locations=[Location(lat=38.98582939, lon=-76.937329584)], interests=['shopping', 'Go Kart', 'Arcade', 'Theme Parks', 'Aquatic Sports', 'Japanese food', 'Museums', 'Football', 'Culture', 'Music']))
        {'recommendations': [{'name': 'Potomac Pizza', 'address': '7777 Baltimore Ave, College Park, MD  20740, United States', 'location': {'lat': 38.9873178, 'lon': -76.9356036}, 'category': 'Restaurant', 'category2': 'Italian food'}, {'name': 'The Spa at The Hotel at the University of Maryland', 'address': '7777 Baltimore Ave, FL 4, College Park, MD  20740, United States', 

        '''
        centroid = self.calculate_centroid(request.locations)
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

    def prepare_data(self, data, name, budget, other=[]):
        '''
        Prepare the data for the model
        Data is the self.locationsList
        '''
        places_df = pd.DataFrame(data)
        user_profile = {
            "Group Name": name,
            "Theme": self.theme,
            "Budget": budget,
            "Other": other
        }
        places_df['combined_category'] = places_df.apply(
            lambda row: [row['category'], row['category2']], axis=1)
        df_user_profile = pd.DataFrame([user_profile])
        df_user_profile['combined_preferences'] = df_user_profile.apply(
            lambda row: row['Theme'] + row['Other'], axis=1)
        mlb = MultiLabelBinarizer()
        le_name = LabelEncoder()
        le_address = LabelEncoder()
        # Fit the MultiLabelBinarizer on both places and user preferences
        all_categories = list(places_df['combined_category'].explode().unique(
        )) + list(df_user_profile['combined_preferences'].explode().unique())
        mlb.fit([all_categories])

        # Transform the combined categories
        places_encoded = mlb.transform(places_df['combined_category'])
        user_encoded = mlb.transform(df_user_profile['combined_preferences'])

        # Encode the name and address columns
        places_df['name_encoded'] = le_name.fit_transform(places_df['name'])
        places_df['address_encoded'] = le_address.fit_transform(
            places_df['address'])

        # Extract latitude and longitude
        places_df['lat'] = places_df['location'].apply(lambda x: x['lat'])
        places_df['lon'] = places_df['location'].apply(lambda x: x['lon'])

        # Create the final places feature matrix
        places_features = np.hstack(
            [places_df[['name_encoded', 'address_encoded', 'lat', 'lon']].values, places_encoded])
        user_features = np.hstack(
            [user_encoded, df_user_profile[['Budget']].values])

        places_features_df = pd.DataFrame(places_features)
        user_features_df = pd.DataFrame(user_features)
        places_tensor = torch.tensor(places_features, dtype=torch.float32)
        user_tensor = torch.tensor(user_features, dtype=torch.float32)
        return places_features_df, user_features_df, places_tensor, user_tensor


async def main():
    users = ["66996b7b0025b402922b", "66996d2e00314baa2a20"]
    locations = [Location(lat=38.98582939, lon=-76.937329584)]

    # model = await AiModel.create(users, locations)

    # print("\nLocations List (JSON):")
    # pprint(type(model.locationsList))  # Parse JSON for prettier printing
    # print(model.locationsList)
    model = AiModel(users, locations, "shopping_spree")
    data = [{'name': 'Potomac Pizza', 'address': '7777 Baltimore Ave, College Park, MD  20740, United States', 'location': {'lat': 38.9873178, 'lon': -76.9356036}, 'category': 'Restaurant', 'category2': 'Italian food'}, {'name': 'Iron Rooster - College Park', 'address': '7777 Baltimore Ave, College Park, MD  20740, United States', 'location': {'lat': 38.9869586, 'lon': -76.935151}, 'category': 'Restaurant', 'category2': 'Eating'}, {'name': 'GrillMarx Steakhouse & Raw Bar', 'address': '7777 Baltimore Ave, College Park, MD  20740, United States', 'location': {'lat': 38.9869586, 'lon': -76.935151}, 'category': 'Restaurant', 'category2': 'Eating'}, {'name': 'The Hall CP', 'address': '4656 Hotel Drive, College Park, MD 20742, United States', 'location': {'lat': 38.9861878, 'lon': -76.9336134}, 'category': 'Restaurant', 'category2': 'Eating'}, {'name': 'Wasabi Bistro Japanese Food & Bubble Tea', 'address': '4505 College Ave, College Park, MD  20740, United States', 'location': {'lat': 38.9818553, 'lon': -76.9373256}, 'category': 'Restaurant', 'category2': 'Japanese food'}, {'name': 'Qù Japan', 'address': '7406 Baltimore Ave, College Park, MD  20740, United States', 'location': {'lat': 38.9811565, 'lon': -76.9380815}, 'category': 'Restaurant', 'category2': 'Japanese food'}, {'name': 'Marathon Deli', 'address': '4429 Lehigh Rd, College Park, MD  20740, United States', 'location': {'lat': 38.9811446, 'lon': -76.9380934}, 'category': 'Restaurant', 'category2': 'Eating'}, {'name': 'College Park Shopping Center', 'address': '7370 Baltimore Ave, College Park, MD  20740, United States', 'location': {'lat': 38.9806676, 'lon': -76.9390872}, 'category': 'Store', 'category2': 'shopping'}, {'name': 'Eastern Gourmet', 'address': '8150 Baltimore Ave, Ste D, College Park, MD  20740, United States', 'location': {'lat': 38.9911763, 'lon': -76.9340584}, 'category': 'Restaurant', 'category2': 'Eating'}, {'name': 'The Spot Mini', 'address': '4207 Knox Rd, College Park, MD  20740, United States', 'location': {'lat': 38.9810829, 'lon': -76.9426016}, 'category': 'Restaurant', 'category2': 'Japanese food'}, {'name': 'Bandit Taco', 'address': '4426 Calvert Rd\nMaryland, MD  20740\nUnited States', 'location': {'lat': 38.9786667, 'lon': -76.9381931}, 'category': 'Restaurant', 'category2': 'Eating'}, {'name': 'Roots Natural Kitchen', 'address': '4420 Calvert Rd, College Park, MD  20740, United States', 'location': {'lat': 38.9786754, 'lon': -76.9385314}, 'category': 'Restaurant', 'category2': 'Eating'}, {'name': 'Adele H. Stamp Student Union', 'address': '3972 Campus Dr, College Park, MD  20742, United States', 'location': {'lat': 38.9880809, 'lon': -76.9448787}, 'category': 'University', 'category2': 'Arcade'}, {'name': 'MeatUp Korean BBQ & Bar', 'address': '8503 Baltimore Ave, College Park, MD  20740, United States', 'location': {'lat': 38.9948577, 'lon': -76.9319577}, 'category': 'Restaurant', 'category2': 'Eating'}, {'name': 'Onikama Ramen Bar', 'address': '3711 Campus Dr, College Park, MD  20740, United States', 'location': {'lat': 38.9840987, 'lon': -76.9493794}, 'category': 'Restaurant', 'category2': 'Japanese food'}, {'name': 'Tacos A La Madre', 'address': '5010 Berwyn Rd, College Park, MD  20740, United States', 'location': {'lat': 38.9948437, 'lon': -76.9258524}, 'category': 'Restaurant', 'category2': 'Eating'}, {'name': 'Horu Sushi Kitchen', 'address': '4407 Woodberry St, Riverdale, MD 20737, United States', 'location': {'lat': 38.970507, 'lon': -76.936748}, 'category': 'Restaurant', 'category2': 'Japanese food'}, {'name': 'College Park Aviation Museum', 'address': '1985 Corporal Frank Scott Dr, College Park, MD 20740, United States', 'location': {'lat': 38.9788133, 'lon': -76.9222505}, 'category': 'Museum', 'category2': 'Museums'}, {'name': 'The Jerk Pit', 'address': '9078 Baltimore Ave, College Park, MD  20740, United States', 'location': {'lat': 39.0026555, 'lon': -76.931175}, 'category': 'Restaurant', 'category2': 'Eating'}, {'name': 'Paint Branch Golf Complex', 'address': '4690 University Blvd, College Park, MD  20740, United States', 'location': {'lat': 39.0038245, 'lon': -76.93561}, 'category': 'Golf', 'category2': 'Golf'}, {'name': 'University of Maryland Golf Course', 'address': '3800 Golf Course Rd, College Park, MD 20742, United States', 'location': {'lat': 38.9910788, 'lon': -76.9548726}, 'category': 'Golf', 'category2': 'Golf'}, {'name': 'Calvert Road Disc Golf Course', 'address': '5201 Paint Branch Pkwy, College Park, MD 20740, United States', 'location': {'lat': 38.9756263, 'lon': -76.9167584}, 'category': 'Golf', 'category2': 'Golf'}, {'name': 'Sushinado & Teriyaki', 'address': '6450 America Blvd, Unit 103, Hyattsville, MD  20782, United States', 'location': {'lat': 38.9677584, 'lon': -76.9522842}, 'category': 'Restaurant', 'category2': 'Japanese food'}, {'name': 'The Shoppes At Metro Station', 'address': '6211 Belcrest Rd, Hyattsville, MD  20782, United States', 'location': {'lat': 38.9655546, 'lon': -76.9532033}, 'category': 'Store', 'category2': 'shopping'}, {'name': "Mall at Prince George's", 'address': '3500 East West Hwy, Hyattsville, MD 20782, United States', 'location': {'lat': 38.9683307, 'lon': -76.9574207}, 'category': 'Store', 'category2': 'shopping'}, {'name': 'Olive Garden Italian Restaurant', 'address': '3480 East-West Hwy, Hyattsville, MD  20782, United States', 'location': {'lat': 38.967275, 'lon': -76.9585524}, 'category': 'Restaurant', 'category2': 'Italian food'}, {'name': 'Mamma Lucia', 'address': '4734 Cherry Hill Rd, College Park, MD  20740, United States', 'location': {'lat': 39.016106, 'lon': -76.928265}, 'category': 'Restaurant', 'category2': 'Italian food'}, {'name': 'Super Japan Japanese Restaurant', 'address': '6096 Greenbelt Rd, Greenbelt, MD  20770, United States', 'location': {'lat': 38.9991254, 'lon': -76.9086206}, 'category': 'Restaurant', 'category2': 'Japanese food'}, {'name': 'Beltway Plaza Mall', 'address': '6000 Greenbelt Rd, Greenbelt, MD 20770, United States', 'location': {'lat': 38.9994255, 'lon': -76.908567}, 'category': 'Store', 'category2': 'shopping'}, {'name': 'Three Brothers Italian Restaurant', 'address': '6160 Greenbelt Rd, Greenbelt, MD  20770, United States', 'location': {'lat': 38.9994867, 'lon': -76.9085112}, 'category': 'Restaurant', 'category2': 'Italian food'}, {'name': 'Laugh Out Loud Stations MEGA Fun Center', 'address': '6250 Greenbelt Rd, Greenbelt, MD  20770, United States', 'location': {'lat': 39.0000933, 'lon': -76.9061712}, 'category': 'Playground', 'category2': 'Arcade'}, {'name': 'Teriyaki express Japanese Grill', 'address': '1425 University Blvd E, Hyattsville, MD  20783, United States', 'location': {'lat': 38.9854885, 'lon': -76.9830082}, 'category': 'Restaurant', 'category2': 'Japanese food'}, {'name': 'Kobe Japan Hibachi and Sushi', 'address': '1163 University Blvd E, Takoma Park, MD  20912, United States', 'location': {'lat': 38.9889565, 'lon': -76.9897038}, 'category': 'Restaurant', 'category2': 'Japanese food'}, {'name': 'Chuck E. Cheese', 'address': '1127 University Blvd E, Takoma Park, MD  20912, United States', 'location': {'lat': 38.9893487, 'lon': -76.9900026}, 'category': 'Restaurant', 'category2': 'Arcade'}, {'name': 'Quickway Japanese Hibachi', 'address': '6300 Annapolis Rd, Unit 7, Hyattsville, MD 20784, United States', 'location': {'lat': 38.940117, 'lon': -76.9066458}, 'category': 'Restaurant', 'category2': 'Japanese food'}, {'name': 'Hip Hop Fish & Chicken', 'address': '5609 Sargent Rd, Hyattsville, MD  20782, United States', 'location': {'lat': 38.9584731, 'lon': -76.9854791}, 'category': 'Restaurant', 'category2': 'British food'}, {'name': 'Hip Hop Fish & Chicken', 'address': '10961 Baltimore Ave, Beltsville, MD  20705, United States', 'location': {'lat': 39.0335897, 'lon': -76.9084566}, 'category': 'Restaurant', 'category2': 'British food'}, {'name': 'Greenway Shopping Center', 'address': '7547 Greenbelt Rd, Greenbelt, MD  20770, United States', 'location': {'lat': 38.9922394, 'lon': -76.8786699}, 'category': 'Store', 'category2': 'shopping'}, {'name': 'Hip Hop Fish & Chicken', 'address': '7524 Annapolis Rd, Hyattsville, MD  20784, United States', 'location': {'lat': 38.9509075, 'lon': -76.8867693}, 'category': 'Restaurant', 'category2': 'British food'}, {'name': 'Shops at Dakota Crossing', 'address': '2438 Market St NE, Washington, DC  20018, United States', 'location': {'lat': 38.9209262, 'lon': -76.9525981}, 'category': 'Store', 'category2': 'shopping'}, {'name': 'Culture', 'address': '2006 Fenwick St NE, Washington, DC  20002, United States', 'location': {'lat': 38.9146324, 'lon': -76.9852431}, 'category': 'Culture', 'category2': 'Culture'}, {'name': 'Hip Hop Fish & Chicken', 'address': '8000 Martin Luther King Jr Hwy, Lanham, MD  20706, United States',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    'location': {'lat': 38.9323073, 'lon': -76.86898}, 'category': 'Restaurant', 'category2': 'British food'}, {'name': 'Citizens & Culture', 'address': '8113 Georgia Ave, Silver Spring, MD  20910, United States', 'location': {'lat': 38.9909441, 'lon': -77.0262427}, 'category': 'Restaurant', 'category2': 'Culture'}, {'name': "Dave & Buster's", 'address': '8661 Colesville Rd, Unit E102, Silver Spring, MD  20910, United States', 'location': {'lat': 38.9970747, 'lon': -77.0266661}, 'category': 'Nightlife', 'category2': 'Arcade'}, {'name': 'Langston Golf Course', 'address': '2600 Benning Rd NE, Washington, DC  20002, United States', 'location': {'lat': 38.9008777, 'lon': -76.9674435}, 'category': 'Golf', 'category2': 'Golf'}, {'name': 'Sligo Creek Golf Course', 'address': '9701 Sligo Creek Pkwy, Silver Spring, MD  20901, United States', 'location': {'lat': 39.0139932, 'lon': -77.0262775}, 'category': 'Golf', 'category2': 'Golf'}, {'name': 'The Queen Vic British Pub', 'address': '1206 H St NE, Washington, DC  20002, United States', 'location': {'lat': 38.9004268, 'lon': -76.9898665}, 'category': 'Nightlife', 'category2': 'British food'}, {'name': 'Gunpowder Golf Course', 'address': '14300 Old Gunpowder Rd, Laurel, MD 20707, United States', 'location': {'lat': 39.0864269, 'lon': -76.9201451}, 'category': 'Golf', 'category2': 'Golf'}, {'name': 'Rock Creek Park Golf Course', 'address': '6100 16th St NW, Washington, DC  20011, United States', 'location': {'lat': 38.966433, 'lon': -77.0407677}, 'category': 'Golf', 'category2': 'Golf'}, {'name': 'The Artemis', 'address': '3605 14th St NW, Washington, DC 20016, United States', 'location': {'lat': 38.93641, 'lon': -77.0325288}, 'category': 'Nightlife', 'category2': 'British food'}, {'name': "Caruso's Grocery", 'address': '914 14th St SE, Washington, DC  20003, United States', 'location': {'lat': 38.879753, 'lon': -76.985214}, 'category': 'Restaurant', 'category2': 'Italian food'}, {'name': "Trattoria Alberto's Italian Cuisine", 'address': '506 Eighth St SE, Washington, DC  20003, United States', 'location': {'lat': 38.8824609, 'lon': -76.994699}, 'category': 'Restaurant', 'category2': 'Italian food'}, {'name': 'RPM Italian', 'address': '650 K Street NW, Washington, DC 20001, United States', 'location': {'lat': 38.9021968, 'lon': -77.0209301}, 'category': 'Restaurant', 'category2': 'Italian food'}, {'name': "Player's Club", 'address': '1400 14th St NW, Washington, DC  20005, United States', 'location': {'lat': 38.9090013, 'lon': -77.0322275}, 'category': 'Nightlife', 'category2': 'Arcade'}, {'name': 'Museum of Illusions', 'address': '927 H St NW, Washington, DC 20001, United States', 'location': {'lat': 38.9000056, 'lon': -77.0245797}, 'category': 'Museum', 'category2': 'Museums'}, {'name': 'CityCenterDC', 'address': '825 Tenth St NW, Washington, DC 20001, United States', 'location': {'lat': 38.9004601, 'lon': -77.0255113}, 'category': 'Store', 'category2': 'shopping'}, {'name': 'Penn Social', 'address': '801 E St NW, Washington, DC  20004, United States', 'location': {'lat': 38.8963137, 'lon': -77.0231366}, 'category': 'Nightlife', 'category2': 'Arcade'}, {'name': 'National Gallery of Art', 'address': '6th and Constitution Ave NW, Washington, DC 20565, United States', 'location': {'lat': 38.8912666, 'lon': -77.0199215}, 'category': 'Landmark', 'category2': 'Museums'}, {'name': 'Grazie Nonna', 'address': '1100 15th St NW, Washington, DC  20005, United States', 'location': {'lat': 38.9042968, 'lon': -77.0352845}, 'category': 'Restaurant', 'category2': 'Italian food'}, {'name': 'National Air and Space Museum', 'address': '6th Street and Independence Ave SW, Washington, DC 20560, United States', 'location': {'lat': 38.8878678, 'lon': -77.0197713}, 'category': 'Museum', 'category2': 'Museums'}, {'name': 'Elephant & Castle', 'address': '1201 Pennsylvania Ave NW, Washington, DC  20004, United States', 'location': {'lat': 38.8957756, 'lon': -77.028415}, 'category': 'Restaurant', 'category2': 'British food'}, {'name': 'Westfield Wheaton', 'address': '11160 Veirs Mill Rd, Wheaton, MD 20902, United States', 'location': {'lat': 39.0371255, 'lon': -77.0552752}, 'category': 'Store', 'category2': 'shopping'}, {'name': 'Museum of the Bible', 'address': '400 Fourth St SW, Washington, DC 20024, United States', 'location': {'lat': 38.8847886, 'lon': -77.0169894}, 'category': 'Museum', 'category2': 'Museums'}, {'name': 'Culture House', 'address': '700 Delaware Ave SW, Washington, DC  20024, United States', 'location': {'lat': 38.8804475, 'lon': -77.0119017}, 'category': 'Culture', 'category2': 'Culture'}, {'name': 'National Museum of Natural History', 'address': '10th St & Constitution Ave NW, Washington, DC 20560, United States', 'location': {'lat': 38.891278, 'lon': -77.025933}, 'category': 'Landmark', 'category2': 'Museums'}, {'name': 'Across The Pond', 'address': '1732 Connecticut Ave NW, Washington, DC  20009, United States', 'location': {'lat': 38.9136053, 'lon': -77.0460677}, 'category': 'Nightlife', 'category2': 'British food'}, {'name': 'National Museum of American History', 'address': '1300 Constitution Ave NW\nWashington, DC 20560\nUnited States', 'location': {'lat': 38.8911255, 'lon': -77.0302535}, 'category': 'Museum', 'category2': 'Museums'}, {'name': 'Enterprise Golf Course', 'address': '2802 Enterprise Rd, Mitchellville, MD 20721, United States', 'location': {'lat': 38.928356, 'lon': -76.81691}, 'category': 'Golf', 'category2': 'Golf'}, {'name': 'National Museum of African American History and Culture', 'address': '1400 Constitution Ave NW, Washington, DC 20560, United States', 'location': {'lat': 38.8910774, 'lon': -77.0327307}, 'category': 'Museum', 'category2': 'Museums'}, {'name': 'Spy Museum', 'address': "700 L'Enfant Plaza SW, Washington, DC  20024, United States", 'location': {'lat': 38.8839761, 'lon': -77.0254362}, 'category': 'Museum', 'category2': 'Museums'}, {'name': 'Artechouse', 'address': '1238 Maryland Ave SW, Washington, DC  20024, United States', 'location': {'lat': 38.8839544, 'lon': -77.0292483}, 'category': 'Museum', 'category2': 'Museums'}, {'name': 'Gordon Ramsay Fish & Chips', 'address': '665 Wharf St SW, Ste 730, Washington, DC  20024, United States', 'location': {'lat': 38.8775368, 'lon': -77.0228344}, 'category': 'Restaurant', 'category2': 'British food'}, {'name': 'The Brighton SW1', 'address': '949 Wharf St SW, Washington, DC  20024, United States', 'location': {'lat': 38.8801253, 'lon': -77.0260705}, 'category': 'Nightlife', 'category2': 'British food'}, {'name': 'Boardwalk Bar & Arcade', 'address': '715 Wharf St SW, Washington, DC  20024, United States', 'location': {'lat': 38.8784704, 'lon': -77.0240762}, 'category': 'Nightlife', 'category2': 'Arcade'}, {'name': 'East Potomac Golf Course', 'address': '972 Ohio Dr SW, Washington, DC  20024, United States', 'location': {'lat': 38.8745842, 'lon': -77.0267934}, 'category': 'Golf', 'category2': 'Golf'}, {'name': 'il Canale', 'address': '1065 31st St NW, Washington, DC  20007, United States', 'location': {'lat': 38.9045021, 'lon': -77.0609403}, 'category': 'Restaurant', 'category2': 'Italian food'}, {'name': 'Filomena Ristorante', 'address': '1063 Wisconsin Ave NW, Washington, DC  20007, United States', 'location': {'lat': 38.9044418, 'lon': -77.0626614}, 'category': 'Restaurant', 'category2': 'Italian food'}, {'name': 'Vr Zone Dc', 'address': '2300 Wisconsin Ave NW, Unit G - 101, Washington, DC 20007, United States', 'location': {'lat': 38.9201458, 'lon': -77.0719399}, 'category': 'Arcade', 'category2': 'Arcade'}, {'name': "Dave & Buster's", 'address': '1851 Ritchie Station Ct, Capitol Heights, MD  20743, United States', 'location': {'lat': 38.8636088, 'lon': -76.8496569}, 'category': 'Nightlife', 'category2': 'Arcade'}, {'name': 'Fashion Centre at Pentagon City', 'address': '1100 S Hayes St, Arlington, VA  22202, United States', 'location': {'lat': 38.8632295, 'lon': -77.0609432}, 'category': 'Store', 'category2': 'shopping'}, {'name': 'Bowie Golf Club', 'address': '7420 Laurel Bowie Rd\nBowie, MD  20715\nUnited States', 'location': {'lat': 38.9960736, 'lon': -76.7622685}, 'category': 'Golf', 'category2': 'Golf'}, {'name': 'Bowie Town Center', 'address': '15606 Emerald Way, Bowie, MD  20716, United States', 'location': {'lat': 38.9442596, 'lon': -76.7347193}, 'category': 'Store', 'category2': 'shopping'}, {'name': 'GameOn bar+arcade', 'address': '6000 Merriweather Dr, Columbia, MD  21044, United States', 'location': {'lat': 39.206706, 'lon': -76.8625447}, 'category': 'Nightlife', 'category2': 'Arcade'}, {'name': 'Foothill St', 'address': 'Foothill St, Woodbridge, VA  22192, United States', 'location': {'lat': 38.691924520977786, 'lon': -77.3075897230227}, 'category': 'Football', 'category2': 'Football'}]
    model.prepare_data(data, 'Group Name', 100, ['Other'])

# Run the async function
asyncio.run(main())
