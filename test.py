from dotenv import load_dotenv
import os
import googlemaps

# Load environment variables
load_dotenv()

# Initialize the Google Maps client with the API key from .env file
api_key = os.getenv('MAPS_API')  # Securely load the API key
gmaps = googlemaps.Client(key=api_key)

# Define the addresses and interests
addresses = [
    '1600 Amphitheatre Parkway, Mountain View, CA',
    'One Apple Park Way, Cupertino, CA',
    '1 Infinite Loop, Cupertino, CA'
]

interests = [
    ['sushi', 'japanese', 'seafood'],
    ['vegan', 'organic', 'healthy'],
    ['pizza', 'italian', 'pasta']
]

# Function to get nearby restaurants based on interests


def get_nearby_restaurants(location, interest_keywords):
    places_result = gmaps.places_nearby(
        location=location, radius=5000, type='restaurant')
    filtered_restaurants = []
    for place in places_result['results']:
        if any(interest in place['name'].lower() or interest in place.get('types', []) for interest in interest_keywords):
            filtered_restaurants.append(place)
    return filtered_restaurants[:5]


# Process each address and interest
for i, address in enumerate(addresses):
    # Geocode the address to get the latitude and longitude
    geocode_result = gmaps.geocode(address)
    location = geocode_result[0]['geometry']['location']
    lat_lng = (location['lat'], location['lng'])

    # Get nearby restaurants based on interests
    nearby_restaurants = get_nearby_restaurants(lat_lng, interests[i])

    # Print the address and nearby restaurants
    print(f"Restaurants near {address} based on interests {interests[i]}:")
    for restaurant in nearby_restaurants:
        name = restaurant['name']
        address = restaurant.get('vicinity', 'No address available')
        rating = restaurant.get('rating', 'No rating available')
        print(f"- {name}, {address}, Rating: {rating}")
    print("\n")
