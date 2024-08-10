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
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from fastapi import HTTPException
from typeDefs import Location


def load_model(file_path, num_users, num_items, new_input_dim):
    model_info = torch.load(file_path)
    old_input_dim = model_info['content_input_dim']
    num_factors = model_info['num_factors']

    # Create a new model with the current input dimension
    new_model = HybridModel(
        content_input_dim=new_input_dim,
        num_users=num_users,
        num_items=num_items,
        num_factors=num_factors
    )

    # Load the saved state dict
    old_state_dict = model_info['state_dict']

    # Create a new state dict, copying over compatible parts
    new_state_dict = {}

    # Handle the content model weights
    new_state_dict['content_model.fc1.weight'] = torch.nn.init.xavier_uniform_(
        torch.empty(128, new_input_dim))
    new_state_dict['content_model.fc1.bias'] = old_state_dict['content_model.fc1.bias']

    # Copy over the rest of the layers
    for k, v in old_state_dict.items():
        if k not in ['content_model.fc1.weight', 'cf_model.user_factors.weight', 'cf_model.item_factors.weight']:
            new_state_dict[k] = v

    # Initialize new embeddings for users and items
    new_state_dict['cf_model.user_factors.weight'] = torch.nn.init.xavier_uniform_(
        torch.empty(num_users, num_factors))
    new_state_dict['cf_model.item_factors.weight'] = torch.nn.init.xavier_uniform_(
        torch.empty(num_items, num_factors))

    # Load the new state dict
    new_model.load_state_dict(new_state_dict)

    return new_model


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


class ProximityRecommendationRequest(BaseModel):
    locations: List[Location]
    interests: List[str]


class ContentModel(nn.Module):
    def __init__(self, input_dim):
        super(ContentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc4(x)
        return x


class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super(CollaborativeFilteringModel, self).__init__()
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)
        # add biases later on

    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)


class HybridModel(nn.Module):
    def __init__(self, content_input_dim, num_users, num_items, num_factors):
        super(HybridModel, self).__init__()
        self.content_model = ContentModel(content_input_dim)
        self.cf_model = CollaborativeFilteringModel(
            num_users, num_items, num_factors)
        self.fc = nn.Linear(2, 1)  # Combine content and CF scores

    def forward(self, content_input, user, item):
        content_score = self.content_model(content_input)
        cf_score = self.cf_model(user, item)
        combined = torch.cat((content_score, cf_score.unsqueeze(1)), dim=1)
        return self.fc(combined)


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

    vacation = [
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
        "vacation": vacation,
        "food_and_drink": food_and_drink
    }

    def __init__(self, users: List[str], location: List[Location], theme: str, other: List[str] = [], budget: int = 100):
        self.users = users
        self.location = location
        self.preferences = None
        self.top_interests = None
        self.locationsList = None
        self.other = other
        self.theme = self.__class__.theme_categories[theme]
        self.places_df = None
        self.user_tensor = None
        self.places_tensor = None
        self.budget = budget
        self.num_users = 0
        self.num_items = 0
        self.recs = None

    def load_model(self, input_dim):
        self.model = load_model(
            "model.pth", self.num_users, self.num_items, input_dim)
        return self.model

    def get_recommendations(self, user_idx):
        self.model.eval()
        with torch.no_grad():
            current_user_tensor = self.user_tensor[user_idx].unsqueeze(0)
            current_user_tensor_repeated = current_user_tensor.repeat(
                self.places_tensor.shape[0], 1)
            user_place_tensor = torch.cat(
                (current_user_tensor_repeated, self.places_tensor), dim=1)

            user_indices = torch.full(
                (self.places_tensor.shape[0],), user_idx, dtype=torch.long)
            item_indices = torch.arange(
                self.places_tensor.shape[0], dtype=torch.long)

            predictions = self.model(
                user_place_tensor, user_indices, item_indices)
            predictions = predictions.numpy().flatten()

        recommendations = self.places_df.copy()
        recommendations['hybrid_score'] = predictions

        content_scores = self.model.content_model(
            user_place_tensor).detach().numpy().flatten()
        cf_scores = self.model.cf_model(
            user_indices, item_indices).detach().numpy().flatten()

        recommendations['content_score'] = content_scores
        recommendations['cf_score'] = cf_scores

        def normalize_score_hybrid(score):
            return 10 * (score - score.min()) / (score.max() - score.min())

        recommendations['hybrid_score'] = normalize_score_hybrid(
            recommendations['hybrid_score'])

        # Sort recommendations by hybrid score
        recommendations = recommendations.sort_values(
            by='hybrid_score', ascending=False)

        return recommendations

    def train_model(self, epochs=1000):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            total_loss = 0
            for user_idx in range(self.user_tensor.shape[0]):
                current_user_tensor = self.user_tensor[user_idx].unsqueeze(0)
                current_user_tensor_repeated = current_user_tensor.repeat(
                    self.places_tensor.shape[0], 1)
                user_place_tensor = torch.cat(
                    (current_user_tensor_repeated, self.places_tensor), dim=1)

                current_user_pref_tensor = torch.tensor(
                    self.user_encoded[user_idx], dtype=torch.float32).unsqueeze(0)
                place_pref_tensor = torch.tensor(
                    self.places_encoded, dtype=torch.float32)
                similarity = torch.tensor(cosine_similarity(
                    current_user_pref_tensor, place_pref_tensor).reshape(-1, 1), dtype=torch.float32)

                optimizer.zero_grad()
                user_indices = torch.full(
                    (self.places_tensor.shape[0],), user_idx, dtype=torch.long)
                item_indices = torch.arange(
                    self.places_tensor.shape[0], dtype=torch.long)
                output = self.model(user_place_tensor,
                                    user_indices, item_indices)
                loss = criterion(output, similarity)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch+1) % 100 == 0:
                print(f'Epoch {
                      epoch+1}/{epochs}, Avg Loss: {total_loss/self.user_tensor.shape[0]:.4f}')

    async def initialize(self):
        # 1. Get preferences
        self.preferences = self.getPreferences(self.users)

        # 2 & 3: Get Top interests
        self.top_interests = self.getTopInterests(self.preferences)
        print("Model")
        print(type(self.location))
        print(type(self.location[0]))

        # 4: Get recommendations
        requestRec = ProximityRecommendationRequest(
            locations=self.location, interests=self.top_interests)
        locs = await self.get_proximity_recommendations(requestRec, self.other)

        # 5: Store recommendations as json
        self.locationsList = (locs['recommendations'])

        # 6: Prepare data for model
        # TODO THIS IS JUST FOR TESTING
        ratings_data_default = {
            "User": ["User1", "User1", "User2", "User3", "User4", "User4", "User3", "User2"],
            "Address": ["Potomac Pizza", "SeoulSpice", "Mamma Lucia", "National Archives archeological site", "Pebbles Wellness Spa", "Looney's Pub", "University of Maryland Golf Course", "The Cornerstone Grill & Loft"],
            "Rating": [5, 2, 1, 2, 2, 3, 2, 2]
        }

        self.prepare_data(self.locationsList, self.users,
                          self.budget, ratings_data_default)

        # Loading model --- Create model, then train
        input_dim = self.places_tensor.shape[1] + self.user_tensor.shape[1]
        self.model = self.load_model(input_dim)

        # Train the model
        self.train_model()

        # 7: Get recommendations and return it
        self.recs = self.get_recommendations(0)

    @classmethod
    async def create(cls, users: List[str], location: List[Location], theme: str, other: List[str] = [], budget: int = 100):
        instance = cls(users, location, theme, other, budget)
        await instance.initialize()
        return instance

    def getPreferences(self, users: List[str]):
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
        latitudes = [loc.lat for loc in locations]
        longitudes = [loc.lon for loc in locations]

        centroid_lat = sum(latitudes) / len(locations)
        centroid_lon = sum(longitudes) / len(locations)

        return {"lat": centroid_lat, "lon": centroid_lon}

    def getTopInterests(self, preferences: Dict[str, int], top_n: int = 7) -> List[str]:
        filtered_preferences = {k: v for k, v in preferences.items() if v > 2}
        sorted_preferences = sorted(
            filtered_preferences.items(), key=lambda item: item[1], reverse=True)

        top_interests = [k for k, v in sorted_preferences[:top_n]]

        if len(top_interests) < top_n:
            remaining_preferences = {
                k: v for k, v in preferences.items() if k not in top_interests}
            additional_interests = random.sample(
                list(remaining_preferences.keys()), top_n - len(top_interests))
            top_interests.extend(additional_interests)

        return top_interests

    async def get_proximity_recommendations(self, request: ProximityRecommendationRequest, other: List[str] = []):
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
            for otherInterest in other:
                results = await apple_maps_service.search(
                    otherInterest, centroid['lat'], centroid['lon'])
                for result in results:
                    if not result['category']:
                        result['category'] = otherInterest
                    result['category2'] = otherInterest
                all_recommendations.extend(results)

            sorted_recommendations = sorted(
                all_recommendations,
                key=lambda x: ((x['location']['lat'] - centroid['lat'])**2 +
                               (x['location']['lon'] - centroid['lon'])**2)**0.5
            )

            return {"recommendations": sorted_recommendations}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error getting recommendations: {str(e)}")

    def prepare_data(self, data, name, budget, ratingsData=[]):
        places_df = pd.DataFrame(data)
        self.places_df = places_df
        user_profile = [{
            "Group Name": name,
            "Theme": self.theme,
            "Budget": budget,
            "Other": self.other
        }]
        df_user_profile = pd.DataFrame(user_profile, index=[0])
        places_df['combined_category'] = places_df.apply(
            lambda row: [row['category'], row['category2']], axis=1)
        df_user_profile['combined_preferences'] = df_user_profile.apply(
            lambda row: row['Theme'] + row['Other'], axis=1)
        mlb = MultiLabelBinarizer()
        le_name = LabelEncoder()
        le_address = LabelEncoder()

        all_categories = list(places_df['combined_category'].explode().unique(
        )) + list(df_user_profile['combined_preferences'].explode().unique())
        mlb.fit([all_categories])

        places_encoded = mlb.transform(places_df['combined_category'])
        user_encoded = mlb.transform(df_user_profile['combined_preferences'])

        places_df['name_encoded'] = le_name.fit_transform(places_df['name'])
        places_df['address_encoded'] = le_address.fit_transform(
            places_df['address'])

        places_df['lat'] = places_df['location'].apply(lambda x: x['lat'])
        places_df['lon'] = places_df['location'].apply(lambda x: x['lon'])

        places_features = np.hstack(
            [places_df[['name_encoded', 'address_encoded', 'lat', 'lon']].values, places_encoded])
        user_features = np.hstack(
            [user_encoded, df_user_profile[['Budget']].values])
        self.places_tensor = torch.tensor(places_features, dtype=torch.float32)
        self.user_tensor = torch.tensor(user_features, dtype=torch.float32)

        ratings_df = pd.DataFrame(ratingsData)
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()

        ratings_df['user_idx'] = user_encoder.fit_transform(ratings_df['User'])
        ratings_df['item_idx'] = item_encoder.fit_transform(
            ratings_df['Address'])

        self.num_users = len(user_encoder.classes_)
        self.num_items = len(places_df)
        interaction_matrix = np.zeros((self.num_users, self.num_items))

        for _, row in ratings_df.iterrows():
            user_idx = row['user_idx']
            item_idx = row['item_idx']
            rating = row['Rating']
            interaction_matrix[user_idx, item_idx] = rating

        interaction_tensor = torch.tensor(
            interaction_matrix, dtype=torch.float32)

        # Store encoded data for training
        self.user_encoded = user_encoded
        self.places_encoded = places_encoded

        return self.places_tensor, self.user_tensor, interaction_tensor


# async def main():
#     users = ["66996b7b0025b402922b", "66996d2e00314baa2a20"]
#     locations = [Location(lat=38.98582939, lon=-76.937329584)]

#     model = await AiModel.create(users, locations, "shopping_spree", ["Japanese food"])
#     print(model.recs)

# # Run the async function
# asyncio.run(main())


# When calling API Requeast take in: userList, locationList as locaiton objects, theme, and comma separated list of "Other"
