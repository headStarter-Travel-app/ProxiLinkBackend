import jwt
import time
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import os
from dotenv import load_dotenv
import requests

token = os.getenv('TOKEN_TEMP')


def get_nearest_locations(addresses, interests):
    url = 'https://api.apple-mapkit.com/v1/nearest-locations'  # Hypothetical endpoint
    headers = {
        'Authorization': f'Bearer {token}',
        'User-Agent': 'your_app_name',
        'Accept': 'application/json'
    }
    params = {
        'addresses': addresses,
        'interests': interests
    }
    response = requests.get(url, headers=headers, params=params)
    return response.json()


# Example addresses and interests
addresses = [
    "5245 East Joppa Road, Perry Hall, MD 21128",
]
interests = ["restaurants", "parks", "museums"]

# Fetch the nearest locations
nearest_locations = get_nearest_locations(addresses, interests)
print(nearest_locations)
