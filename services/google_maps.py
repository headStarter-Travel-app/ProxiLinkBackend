import requests
import os
import json
from dotenv import load_dotenv


class GoogleMapsService:
    def __init__(self):
        # self.api_key1 = os.getenv("GOOGLE_MAPS_API")
        # self.api_key2 = os.getenv("GOOGLE_MAPS_API2")
        # self.current_key_index = 0
        self.api_keys = []
        for i in range(1, 8):  # Adjuast number basedon API keys
            key = os.getenv(f"GOOGLE_MAPS_API{i}")
            if key:
                self.api_keys.append(key)
        self.current_key_index = 0

    def get_next_api_key(self):
        # if self.current_key_index == 0:
        #     self.current_key_index = 1
        #     return self.api_key1
        # else:
        #     self.current_key_index = 0
        #     return self.api_key2
        api_key = self.api_keys[self.current_key_index]
        self.current_key_index = (
            self.current_key_index + 1) % len(self.api_keys)
        return api_key

    def find_place(self, input_text):
        api_key = self.get_next_api_key()
        # print(api_key)
        # Print the API key being used COMMENT OUT

        base_url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-FieldMask": "places.displayName,places.id,places.formattedAddress,places.photos,places.currentOpeningHours,places.rating,places.websiteUri,places.priceLevel",
            'X-Goog-Api-Key': api_key
        }
        data = {
            "textQuery": input_text,
            "languageCode": "en"
        }

        response = requests.post(base_url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            self.add_photo_urls(result, api_key)
            return result
        else:
            return f"Error: {response.status_code}, {response.text}"

    def add_photo_urls(self, result, api_key):
        if 'places' in result:
            for place in result['places']:
                if 'photos' in place:
                    photo_urls = []
                    for photo in place['photos'][:5]:  # Limit to first 5 photos
                        if 'name' in photo:
                            photo_url = f"https://places.googleapis.com/v1/{
                                photo['name']}/media?key={api_key}&maxHeightPx=400&maxWidthPx=400"
                            photo_urls.append(photo_url)
                    place['photo_urls'] = photo_urls


# Usage example
load_dotenv()  # Load environment variables from .env file
google_maps_service = GoogleMapsService()


# for _ in range(8):
#     res = google_maps_service.find_place(
#         "1881 Post St, San Francisco, CA  94115, United States AMC Kabuki ")
#     print(res)
