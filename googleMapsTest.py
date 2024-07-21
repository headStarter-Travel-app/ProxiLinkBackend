import requests
import os
import json
from dotenv import load_dotenv


def find_place(api_key, input_text, input_type="textquery"):
    base_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": input_text,
        "inputtype": input_type,
        "fields": "name,formatted_address,photos,opening_hours,rating",
        "key": api_key
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.text}"


# Usage example
load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("GOOGLE_MAPS_API")
place_to_find = 'Dunkin Honeygo Village Center, 5003 Honeygo Center Dr, Perry Hall, MD 21128'

result = find_place(api_key, place_to_find)
print(json.dumps(result, indent=2))
