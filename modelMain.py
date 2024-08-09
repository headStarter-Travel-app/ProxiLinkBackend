#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


# In[32]:


data = [
    {
        "name": "Tulio",
        "address": "1100 5th Ave, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.6074618,
            "lon": -122.3323898
        },
        "category": "Restaurant",
        "category2": "Italian food"
    },
    {
        "name": "Owl 'N Thistle",
        "address": "808 Post Ave, Seattle, WA  98104, United States",
        "location": {
            "lat": 47.6033029,
            "lon": -122.3356745
        },
        "category": "Restaurant",
        "category2": "Nightlife"
    },
    {
        "name": "Owl 'N Thistle",
        "address": "808 Post Ave, Seattle, WA  98104, United States",
        "location": {
            "lat": 47.6033029,
            "lon": -122.3356745
        },
        "category": "Restaurant",
        "category2": "Bars"
    },
    {
        "name": "Benaroya Hall",
        "address": "200 University St, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.60795,
            "lon": -122.336361
        },
        "category": "MusicVenue",
        "category2": "Music"
    },
    {
        "name": "Cortina",
        "address": "621 Union St, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.610263,
            "lon": -122.3329177
        },
        "category": "Restaurant",
        "category2": "Italian food"
    },
    {
        "name": "Trinity",
        "address": "107 Occidental Ave S, Seattle, WA  98104, United States",
        "location": {
            "lat": 47.6014103,
            "lon": -122.3333859
        },
        "category": "Nightlife",
        "category2": "Nightlife"
    },
    {
        "name": "Skalka",
        "address": "77 Spring St, Seattle, WA  98104, United States",
        "location": {
            "lat": 47.605015,
            "lon": -122.33758
        },
        "category": "Restaurant",
        "category2": "Eating"
    },
    {
        "name": "Lune Cafe",
        "address": "107 1st Ave S, Seattle, WA  98104, United States",
        "location": {
            "lat": 47.6013996,
            "lon": -122.3345108
        },
        "category": "Cafe",
        "category2": "Eating"
    },
    {
        "name": "The Diller Room",
        "address": "1224 1st Ave, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.606769,
            "lon": -122.3379075
        },
        "category": "Nightlife",
        "category2": "Nightlife"
    },
    {
        "name": "The Diller Room",
        "address": "1224 1st Ave, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.606769,
            "lon": -122.3379075
        },
        "category": "Nightlife",
        "category2": "Bars"
    },
    {
        "name": "Alder & Ash",
        "address": "629 Pike St, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.6112081,
            "lon": -122.33373
        },
        "category": "Restaurant",
        "category2": "Eating"
    },
    {
        "name": "Wild Ginger Kitchen",
        "address": "1401 3rd Ave, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.6088308,
            "lon": -122.337421
        },
        "category": "Restaurant",
        "category2": "Eating"
    },
    {
        "name": "Muse Lounge",
        "address": "224 Occidental Ave S, Seattle, WA  98104, United States",
        "location": {
            "lat": 47.6002453,
            "lon": -122.332558
        },
        "category": "Nightlife",
        "category2": "Nightlife"
    },
    {
        "name": "Emerald City Guitars",
        "address": "83 S Washington St, Seattle, WA  98104, United States",
        "location": {
            "lat": 47.6007267,
            "lon": -122.3351777
        },
        "category": "Store",
        "category2": "Music"
    },
    {
        "name": "Pioneer Square Habitat Beach",
        "address": "99 S Main St, Seattle, WA  98104, United States",
        "location": {
            "lat": 47.6012745,
            "lon": -122.3362719
        },
        "category": "Beach",
        "category2": "Beach"
    },
    {
        "name": "Luigi's Italian Eatery",
        "address": "211 1st Ave S, Seattle, WA  98104, United States",
        "location": {
            "lat": 47.6004215,
            "lon": -122.3345818
        },
        "category": "Restaurant",
        "category2": "Italian food"
    },
    {
        "name": "Yard House",
        "address": "1501 4th Ave, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.6105718,
            "lon": -122.3372196
        },
        "category": "Nightlife",
        "category2": "Eating"
    },
    {
        "name": "Xtadium Lounge",
        "address": "315 2nd Ave S, Seattle, WA  98104, United States",
        "location": {
            "lat": 47.5995455,
            "lon": -122.3317615
        },
        "category": "Nightlife",
        "category2": "Nightlife"
    },
    {
        "name": "Xtadium Lounge",
        "address": "315 2nd Ave S, Seattle, WA  98104, United States",
        "location": {
            "lat": 47.5995455,
            "lon": -122.3317615
        },
        "category": "Nightlife",
        "category2": "Bars"
    },
    {
        "name": "Ivar's Acres of Clams",
        "address": "1001 Alaskan Way, Seattle, WA  98104, United States",
        "location": {
            "lat": 47.6041059,
            "lon": -122.3391666
        },
        "category": "Restaurant",
        "category2": "Eating"
    },
    {
        "name": "The Showbox",
        "address": "1426 1st Ave, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.6084488,
            "lon": -122.3392653
        },
        "category": "MusicVenue",
        "category2": "Music"
    },
    {
        "name": "Pacific Place",
        "address": "600 Pine St, Seattle, WA 98101, United States",
        "location": {
            "lat": 47.6128596,
            "lon": -122.3351734
        },
        "category": "Store",
        "category2": "shopping"
    },
    {
        "name": "Nordstrom Rack",
        "address": "400 Pine St, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.6118329,
            "lon": -122.3373005
        },
        "category": "Store",
        "category2": "shopping"
    },
    {
        "name": "Westlake Center",
        "address": "400 Pine St, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.6119182,
            "lon": -122.3373191
        },
        "category": "Store",
        "category2": "shopping"
    },
    {
        "name": "Luke's Lobster",
        "address": "110 Pike St, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.6091527,
            "lon": -122.3396645
        },
        "category": "Restaurant",
        "category2": "Eating"
    },
    {
        "name": "The Paramount Theatre",
        "address": "911 Pine St, Seattle, WA 98101, United States",
        "location": {
            "lat": 47.6135481,
            "lon": -122.3315406
        },
        "category": "Theater",
        "category2": "Music"
    },
    {
        "name": "Pasta Casalinga",
        "address": "93 Pike St, Unit 201, Seattle, WA 98101, United States",
        "location": {
            "lat": 47.6083837,
            "lon": -122.3400378
        },
        "category": "Restaurant",
        "category2": "Italian food"
    },
    {
        "name": "Il Terrazzo Carmine",
        "address": "411 1st Ave S, Seattle, WA 98104, United States",
        "location": {
            "lat": 47.5986563,
            "lon": -122.3346336
        },
        "category": "Restaurant",
        "category2": "Italian food"
    },
    {
        "name": "Cowgirls Inc.",
        "address": "421 1st Ave S, Seattle, WA  98104, United States",
        "location": {
            "lat": 47.598522,
            "lon": -122.3345922
        },
        "category": "Nightlife",
        "category2": "Nightlife"
    },
    {
        "name": "IL Bistro",
        "address": "93 Pike St, Unit A, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.608654,
            "lon": -122.3402049
        },
        "category": "Restaurant",
        "category2": "Italian food"
    },
    {
        "name": "Alibi Room",
        "address": "85 Pike St, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.6082318,
            "lon": -122.3403704
        },
        "category": "Restaurant",
        "category2": "Nightlife"
    },
    {
        "name": "Alibi Room",
        "address": "85 Pike St, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.6082318,
            "lon": -122.3403704
        },
        "category": "Restaurant",
        "category2": "Eating"
    },
    {
        "name": "The Crab Pot",
        "address": "1301 Alaskan Way\nSeattle, WA 98101\nUnited States",
        "location": {
            "lat": 47.6059677,
            "lon": -122.3409551
        },
        "category": "Restaurant",
        "category2": "Eating"
    },
    {
        "name": "Fog Room",
        "address": "1610 2nd Ave, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.6109298,
            "lon": -122.3399681
        },
        "category": "Nightlife",
        "category2": "Bars"
    },
    {
        "name": "Holy Cow Records",
        "address": "1501 Pike Pl, Unit 325, Seattle, WA 98101, United States",
        "location": {
            "lat": 47.608709,
            "lon": -122.3410473
        },
        "category": "Store",
        "category2": "Music"
    },
    {
        "name": "Sluggers",
        "address": "538 1st Ave S, Seattle, WA  98104, United States",
        "location": {
            "lat": 47.5971473,
            "lon": -122.3334954
        },
        "category": "Nightlife",
        "category2": "Bars"
    },
    {
        "name": "Zig Zag Caf\u00e9",
        "address": "1501 Western Ave, Unit 202, Seattle, WA 98101, United States",
        "location": {
            "lat": 47.6083891,
            "lon": -122.3415828
        },
        "category": "Nightlife",
        "category2": "Bars"
    },
    {
        "name": "Ludi's Restaurant",
        "address": "120 Stewart St, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.6109189,
            "lon": -122.3407888
        },
        "category": "Restaurant",
        "category2": "Eating"
    },
    {
        "name": "The Nest",
        "address": "110 Stewart St, Seattle, WA  98101, United States",
        "location": {
            "lat": 47.610588,
            "lon": -122.3412636
        },
        "category": "Nightlife",
        "category2": "Nightlife"
    },
    {
        "name": "Assaggio Ristorante",
        "address": "2010 4th Ave, Seattle, WA  98121, United States",
        "location": {
            "lat": 47.6134452,
            "lon": -122.3400866
        },
        "category": "Restaurant",
        "category2": "Italian food"
    },
    {
        "name": "The Pink Door",
        "address": "1919 Post Alley, Seattle, WA 98101, United States",
        "location": {
            "lat": 47.6103408,
            "lon": -122.3425869
        },
        "category": "Restaurant",
        "category2": "Bars"
    },
    {
        "name": "The Pink Door",
        "address": "1919 Post Alley, Seattle, WA 98101, United States",
        "location": {
            "lat": 47.6103408,
            "lon": -122.3425869
        },
        "category": "Restaurant",
        "category2": "Italian food"
    },
    {
        "name": "Tavol\u00e0ta",
        "address": "501 E Pike St, Seattle, WA  98122, United States",
        "location": {
            "lat": 47.6139133,
            "lon": -122.3252803
        },
        "category": "Restaurant",
        "category2": "Italian food"
    },
    {
        "name": "Dimitriou's Jazz Alley",
        "address": "2033 6th Ave, Seattle, WA  98121, United States",
        "location": {
            "lat": 47.6148354,
            "lon": -122.3397052
        },
        "category": "MusicVenue",
        "category2": "Music"
    },
    {
        "name": "Deep Dive",
        "address": "620 Lenora St, Seattle, WA  98121, United States",
        "location": {
            "lat": 47.6157612,
            "lon": -122.3392117
        },
        "category": "Nightlife",
        "category2": "Bars"
    },
    {
        "name": "Kpop Nara",
        "address": "501 E Pine St, Seattle, WA  98122, United States",
        "location": {
            "lat": 47.615056,
            "lon": -122.3252696
        },
        "category": "Store",
        "category2": "Music"
    },
    {
        "name": "Sub Pop on 7th",
        "address": "2130 7th Ave, Seattle, WA  98119, United States",
        "location": {
            "lat": 47.6166835,
            "lon": -122.3395999
        },
        "category": "Store",
        "category2": "Music"
    },
    {
        "name": "WAMU Theater",
        "address": "800 Occidental Ave S, Seattle, WA  98134, United States",
        "location": {
            "lat": 47.5932431,
            "lon": -122.3329177
        },
        "category": "Music",
        "category2": "Music"
    },
    {
        "name": "Jupiter Bar",
        "address": "2126 2nd Ave, Unit A, Seattle, WA  98121, United States",
        "location": {
            "lat": 47.6132588,
            "lon": -122.3438155
        },
        "category": "Nightlife",
        "category2": "Bars"
    },
    {
        "name": "Harvard Market",
        "address": "1401 Broadway, Seattle, WA  98122, United States",
        "location": {
            "lat": 47.6133245,
            "lon": -122.3210338
        },
        "category": "Store",
        "category2": "shopping"
    },
    {
        "name": "La Fontana Siciliana",
        "address": "120 Blanchard St, Seattle, WA  98121, United States",
        "location": {
            "lat": 47.6130966,
            "lon": -122.3447938
        },
        "category": "Restaurant",
        "category2": "Italian food"
    },
    {
        "name": "Q Nightclub",
        "address": "1426 Broadway, Seattle, WA  98122, United States",
        "location": {
            "lat": 47.6137221,
            "lon": -122.3205989
        },
        "category": "Nightlife",
        "category2": "Nightlife"
    },
    {
        "name": "Neighbours Nightclub",
        "address": "1509 Broadway, Seattle, WA  98122, United States",
        "location": {
            "lat": 47.6143812,
            "lon": -122.32096
        },
        "category": "Nightlife",
        "category2": "Nightlife"
    },
    {
        "name": "Singles Going Steady",
        "address": "2219 2nd Ave, Unit C, Seattle, WA 98121, United States",
        "location": {
            "lat": 47.6135876,
            "lon": -122.3452692
        },
        "category": "Store",
        "category2": "Music"
    },
    {
        "name": "South Lake Union",
        "address": "333 Boren Ave N, Seattle, WA  98109, United States",
        "location": {
            "lat": 47.621203,
            "lon": -122.3362319
        },
        "category": "Store",
        "category2": "shopping"
    },
    {
        "name": "Cha Cha Lounge",
        "address": "1013 E Pike St, Seattle, WA  98122, United States",
        "location": {
            "lat": 47.613989,
            "lon": -122.3187396
        },
        "category": "Nightlife",
        "category2": "Bars"
    },
    {
        "name": "Chophouse Row",
        "address": "1424 11th Ave, Seattle, WA  98122, United States",
        "location": {
            "lat": 47.6136489,
            "lon": -122.3177305
        },
        "category": "Store",
        "category2": "shopping"
    },
    {
        "name": "Museum of Pop Culture",
        "address": "325 5th Avenue N, Seattle, WA 98109, United States",
        "location": {
            "lat": 47.6214722,
            "lon": -122.3481686
        },
        "category": "Museum",
        "category2": "Culture"
    },
    {
        "name": "Pocket Beach",
        "address": "3131 Elliott Ave, Seattle, WA  98121, United States",
        "location": {
            "lat": 47.617188,
            "lon": -122.3582875
        },
        "category": "Beach",
        "category2": "Beach"
    },
    {
        "name": "The Marketplace at Queen Anne",
        "address": "100 Mercer St, Seattle, WA  98109, United States",
        "location": {
            "lat": 47.6247762,
            "lon": -122.3547865
        },
        "category": "Store",
        "category2": "shopping"
    },
    {
        "name": "Colman Beach",
        "address": "1732 Lakeside Ave S, Seattle, WA  98144, United States",
        "location": {
            "lat": 47.5866502,
            "lon": -122.2863175
        },
        "category": "Beach",
        "category2": "Beach"
    },
    {
        "name": "Mount Baker Beach",
        "address": "2521 Lake Park Dr S, Seattle, WA  98144, United States",
        "location": {
            "lat": 47.5834285,
            "lon": -122.2875045
        },
        "category": "Beach",
        "category2": "Beach"
    },
    {
        "name": "Burke Museum",
        "address": "4303 Memorial Way, Seattle, WA 98195, United States",
        "location": {
            "lat": 47.6604305,
            "lon": -122.3114635
        },
        "category": "Museum",
        "category2": "Culture"
    },
    {
        "name": "Madison Park Beach",
        "address": "4201 E Madison St, Seattle, WA  98112, United States",
        "location": {
            "lat": 47.6351112,
            "lon": -122.2765982
        },
        "category": "Beach",
        "category2": "Beach"
    },
    {
        "name": "University Village",
        "address": "2623 NE University Village St, Seattle, WA 98105, United States",
        "location": {
            "lat": 47.6624971,
            "lon": -122.299118
        },
        "category": "Store",
        "category2": "shopping"
    },
    {
        "name": "Alki Beach Park",
        "address": "2665 Alki Ave SW, Seattle, WA 98116, United States",
        "location": {
            "lat": 47.5802602,
            "lon": -122.4083912
        },
        "category": "Park",
        "category2": "Beach"
    },
    {
        "name": "Pritchard Island Beach",
        "address": "8400 55th Ave S, Seattle, WA  98118, United States",
        "location": {
            "lat": 47.5295302,
            "lon": -122.2630388
        },
        "category": "Beach",
        "category2": "Beach"
    },
    {
        "name": "Matthews Beach Park",
        "address": "5100 NE 93rd St, Seattle, WA 98115, United States",
        "location": {
            "lat": 47.6957109,
            "lon": -122.2726393
        },
        "category": "Park",
        "category2": "Beach"
    },
    {
        "name": "Groveland Beach",
        "address": "SE 58th St And 80th Ave SE, Mercer Island, WA 98040, United States",
        "location": {
            "lat": 47.5511513,
            "lon": -122.2343695
        },
        "category": "Beach",
        "category2": "Beach"
    },
    {
        "name": "The Road-End Beach",
        "address": "9015 NE 47th St, Kirkland, WA  98004, United States",
        "location": {
            "lat": 47.6518607,
            "lon": -122.2181117
        },
        "category": "Beach",
        "category2": "Beach"
    },
    {
        "name": "Third Culture Coffee",
        "address": "80 102nd Ave NE, Bellevue, WA  98004, United States",
        "location": {
            "lat": 47.610828,
            "lon": -122.203842
        },
        "category": "Cafe",
        "category2": "Culture"
    },
    {
        "name": "The Bellevue Collection",
        "address": "575 Bellevue Square, Bellevue, WA 98004, United States",
        "location": {
            "lat": 47.615971,
            "lon": -122.203803
        },
        "category": "Store",
        "category2": "shopping"
    },
    {
        "name": "Westfield Southcenter",
        "address": "2800 Southcenter Mall, Seattle, WA 98188, United States",
        "location": {
            "lat": 47.4588386,
            "lon": -122.2582841
        },
        "category": "Store",
        "category2": "shopping"
    },
    {
        "name": "Fritz",
        "address": "435 Pacific Ave, Bremerton, WA  98337, United States",
        "location": {
            "lat": 47.5664286,
            "lon": -122.627124
        },
        "category": "Restaurant",
        "category2": "Belgian food"
    }
]


places_df = pd.DataFrame(data)
places_df
original_data = places_df.copy()
original_data


# In[33]:


#Themes
# Themes with relevant categories

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



# In[34]:


user_profile = [
    {
        "User": "User1",
        "Theme": relaxation_and_wellness,
        "Other": ["Korean food", "Spa", "Nightlife", "Belgian food"],
        "Budget": 2
    },

]

df_user_profile = pd.DataFrame(user_profile)

df_user_profile


# In[35]:


places_df['combined_category'] = places_df.apply(lambda row: [row['category'], row['category2']], axis=1)
df_user_profile['combined_preferences'] = df_user_profile.apply(lambda row: row['Theme'] + row['Other'], axis=1)

places_df
df_user_profile


# In[36]:


mlb = MultiLabelBinarizer()
le_name = LabelEncoder()
le_address = LabelEncoder()

# Fit the MultiLabelBinarizer on both places and user preferences
all_categories = list(places_df['combined_category'].explode().unique()) + list(df_user_profile['combined_preferences'].explode().unique())
mlb.fit([all_categories])

# Transform the combined categories
places_encoded = mlb.transform(places_df['combined_category'])
user_encoded = mlb.transform(df_user_profile['combined_preferences'])

# Encode the name and address columns
places_df['name_encoded'] = le_name.fit_transform(places_df['name'])
places_df['address_encoded'] = le_address.fit_transform(places_df['address'])

# Extract latitude and longitude
places_df['lat'] = places_df['location'].apply(lambda x: x['lat'])
places_df['lon'] = places_df['location'].apply(lambda x: x['lon'])

# Create the final places feature matrix
places_features = np.hstack([places_df[['name_encoded', 'address_encoded', 'lat', 'lon']].values, places_encoded])
user_features = np.hstack([user_encoded, df_user_profile[['Budget']].values])

places_features


# In[37]:


places_tensor = torch.tensor(places_features, dtype=torch.float32)
user_tensor = torch.tensor(user_features, dtype=torch.float32)
places_tensor, user_tensor

places_tensor.shape, user_tensor.shape


# In[38]:


ratings_data = {
    "User": ["User1", "User1", "User2", "User3", "User4", "User4", "User3", "User2"],
    "Address": ["Potomac Pizza", "SeoulSpice", "Mamma Lucia", "National Archives archeological site", "Pebbles Wellness Spa", "Looney's Pub", "University of Maryland Golf Course", "The Cornerstone Grill & Loft"],
    "Rating": [5, 2, 1, 2, 2, 3, 2, 2]
}

ratings_df = pd.DataFrame(ratings_data)

# # Create user and item indices
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

ratings_df['user_idx'] = user_encoder.fit_transform(ratings_df['User'])
ratings_df['item_idx'] = item_encoder.fit_transform(ratings_df['Address'])

# Create the user-item interaction matrix
num_users = len(user_encoder.classes_)
num_items = len(places_df)
interaction_matrix = np.zeros((num_users, num_items))

for _, row in ratings_df.iterrows():
    user_idx = row['user_idx']
    item_idx = row['item_idx']
    rating = row['Rating']
    interaction_matrix[user_idx, item_idx] = rating

# Convert interaction matrix to tensor
interaction_tensor = torch.tensor(interaction_matrix, dtype=torch.float32)

interaction_tensor
ratings_df


# In[39]:


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


# In[40]:


class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super(CollaborativeFilteringModel, self).__init__()
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)
        
    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)


# In[41]:


class HybridModel(nn.Module):
    def __init__(self, content_input_dim, num_users, num_items, num_factors):
        super(HybridModel, self).__init__()
        self.content_model = ContentModel(content_input_dim)
        self.cf_model = CollaborativeFilteringModel(num_users, num_items, num_factors)
        self.fc = nn.Linear(2, 1)  # Combine content and CF scores
        
    def forward(self, content_input, user, item):
        content_score = self.content_model(content_input)
        cf_score = self.cf_model(user, item)
        combined = torch.cat((content_score, cf_score.unsqueeze(1)), dim=1)
        return self.fc(combined)

# Calculate similarity (cosine similarity)
def cosine_similarity(tensor1, tensor2):
    dot_product = torch.sum(tensor1 * tensor2, dim=1)
    norm1 = torch.norm(tensor1, dim=1)
    norm2 = torch.norm(tensor2, dim=1)
    return dot_product / (norm1 * norm2)


# In[42]:


num_factors = 20
input_dim = places_tensor.shape[1] + user_tensor.shape[1]
model = HybridModel(input_dim, num_users, num_items, num_factors)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model


# In[43]:


epochs = 1000
for epoch in range(epochs):
    total_loss = 0
    for user_idx in range(user_tensor.shape[0]):
        current_user_tensor = user_tensor[user_idx].unsqueeze(0)
        current_user_tensor_repeated = current_user_tensor.repeat(places_tensor.shape[0], 1)
        user_place_tensor = torch.cat((current_user_tensor_repeated, places_tensor), dim=1)
        
        current_user_pref_tensor = torch.tensor(user_encoded[user_idx], dtype=torch.float32).unsqueeze(0)
        place_pref_tensor = torch.tensor(places_encoded, dtype=torch.float32)
        similarity = cosine_similarity(current_user_pref_tensor, place_pref_tensor).reshape(-1, 1)
        
        optimizer.zero_grad()
        user_indices = torch.full((places_tensor.shape[0],), user_idx, dtype=torch.long)
        item_indices = torch.arange(places_tensor.shape[0], dtype=torch.long)
        output = model(user_place_tensor, user_indices, item_indices)
        loss = criterion(output, similarity)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/user_tensor.shape[0]:.4f}')


# In[ ]:





# In[ ]:





# In[44]:


def get_recommendations(user_idx):
    model.eval()
    with torch.no_grad():
        current_user_tensor = user_tensor[user_idx].unsqueeze(0)
        current_user_tensor_repeated = current_user_tensor.repeat(places_tensor.shape[0], 1)
        user_place_tensor = torch.cat((current_user_tensor_repeated, places_tensor), dim=1)
        
        user_indices = torch.full((places_tensor.shape[0],), user_idx, dtype=torch.long)
        item_indices = torch.arange(places_tensor.shape[0], dtype=torch.long)
        
        predictions = model(user_place_tensor, user_indices, item_indices)
        predictions = predictions.numpy().flatten()

    recommendations = places_df.copy()
    recommendations['hybrid_score'] = predictions

    # Get content-based and collaborative filtering scores separately
    content_scores = model.content_model(user_place_tensor).detach().numpy().flatten()
    cf_scores = model.cf_model(user_indices, item_indices).detach().numpy().flatten()

    recommendations['content_score'] = content_scores
    recommendations['cf_score'] = cf_scores

    def normalize_score_hybrid(score):
        return 10 * (score - score.min()) / (score.max() - score.min())

    recommendations['hybrid_score'] = normalize_score_hybrid(recommendations['hybrid_score'])

    # Sort recommendations by hybrid score
    recommendations = recommendations.sort_values(by='hybrid_score', ascending=False)
    

    return recommendations

# Get recommendations for a specific user (e.g., user index 1, which corresponds to "User2")
user_idx = 0
recommendations = get_recommendations(user_idx)
selected_columns = ['name', 'address', 'combined_category', 'hybrid_score', 'content_score', 'cf_score']
recommendations[selected_columns]
# recommendations['name', 'address', 'combined_category', 'hybrid_score', 'content_score', 'cf_score']


# In[45]:


import torch

torch.save(model.state_dict(), 'model.pth')

