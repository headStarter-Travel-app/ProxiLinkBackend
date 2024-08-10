# import requests
# import os
# import json
# from dotenv import load_dotenv
# from services.google_maps import google_maps_service


# def find_place(name, address_input):
#     input = f"{name}, {address_input}"
#     details = google_maps_service.find_place(input)
#     print(details)
#     if 'places' in details and len(details['places']) > 0:
#         place = details['places'][0]
#         place_details = {
#             'address': address_input.address,
#             'ID': place.get('id', '0'),
#             'rating': place.get('rating', 0),
#             'name': place.get('displayName', {}).get('text', 'Default Name'),
#             'hours': place.get('currentOpeningHours', {}).get('weekdayDescriptions', []),
#             'url': place.get('websiteUri', ''),
#             'photos': place.get('photo_urls', [])
#         }
#         print(place_details)

#         logger.info("Storing new details in database")
#         update = database.create_document(
#             database_id=appwrite_config['database_id'],
#             collection_id=appwrite_config['locations_collection_id'],
#             document_id=ID.unique(),
#             data=place_details.model_dump()
#         )
#         return place_details


# # Usage example
# load_dotenv()  # Load environment variables from .env file
# api_key = os.getenv("GOOGLE_MAPS_API1")
# place_to_find = 'Dunkin Honeygo Village Center, 5003 Honeygo Center Dr, Perry Hall, MD 21128'

# result = find_place(api_key, place_to_find)
# print(json.dumps(result, indent=2))
