import googlemaps
import json

# Replace 'YOUR_API_KEY' with your actual Google Maps API key
API_KEY = 'AIzaSyAPWmKvpyjbPAdWK3nyxruJsB4WgQd8kXs'

# Initialize the client
gmaps = googlemaps.Client(key=API_KEY)


def get_place_details(address):
    # Geocode the address to get place ID
    geocode_result = gmaps.geocode(address)
    if not geocode_result:
        print("Address not found.")
        return

    place_id = geocode_result[0]['place_id']

    # Fetch place details using place ID
    place_details = gmaps.place(place_id=place_id)

    if place_details['status'] != 'OK':
        print("Failed to fetch place details.")
        return

    result = place_details['result']
    print(result)

    # Extracting details
    name = result.get('name')
    formatted_address = result.get('formatted_address')
    opening_hours = result.get('opening_hours', {}).get('weekday_text', 'N/A')
    price_level = result.get('price_level', 'N/A')
    reviews = result.get('reviews', [])
    photos = result.get('photos', [])

    # Print place details
    print(f"Name: {name}")
    print(f"Address: {formatted_address}")
    print(f"Opening Hours: {opening_hours}")
    print(f"Price Level: {price_level}")

    print("Reviews:")
    for review in reviews[:5]:  # Limit to first 5 reviews
        print(f"- {review['text']}")

    # Print photos URLs
    print("Photos URLs:")
    for photo in photos[:5]:  # Limit to first 5 photos
        photo_reference = photo['photo_reference']
        photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={
            photo_reference}&key={API_KEY}"
        print(f"- {photo_url}")


if __name__ == "__main__":
    address = input("Enter the address: ")
    get_place_details(address)
