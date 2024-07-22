import requests
import os
import json
from dotenv import load_dotenv


class GoogleMapsService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API")

    def find_place(self, input_text):
        base_url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-FieldMask": "places.displayName,places.id,places.formattedAddress,places.photos,places.currentOpeningHours,places.rating,places.websiteUri,places.priceLevel",
            'X-Goog-Api-Key': self.api_key
        }
        data = {
            "textQuery": input_text,
            "languageCode": "en"
        }

        response = requests.post(base_url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            self.add_photo_urls(result)
            return result
        else:
            return f"Error: {response.status_code}, {response.text}"

    def add_photo_urls(self, result):
        if 'places' in result:
            for place in result['places']:
                if 'photos' in place:
                    photo_urls = []
                    for photo in place['photos'][5:]:  # Limit to first 5 photos
                        if 'name' in photo:
                            photo_url = f"https://places.googleapis.com/v1/{photo['name']}/media?key={
                                self.api_key}&maxHeightPx=400&maxWidthPx=400"
                            photo_urls.append(photo_url)
                    place['photo_urls'] = photo_urls


# Usage example
load_dotenv()  # Load environment variables from .env file
google_maps_service = GoogleMapsService()

# place_to_find = 'The Melt, 925 Market St, San Francisco, CA  94103, United States'

# result = google_maps_service.find_place(place_to_find)
# print(json.dumps(result, indent=2))
