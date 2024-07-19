from typing import List, Dict
from geopy.distance import great_circle
import math


def calculate_centroid(locations: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate the centroid of given locations.
    """
    latitudes = [loc['lat'] for loc in locations]
    longitudes = [loc['lon'] for loc in locations]

    centroid_lat = sum(latitudes) / len(locations)
    centroid_lon = sum(longitudes) / len(locations)

    return {"lat": centroid_lat, "lon": centroid_lon}


def get_search_region(centroid: Dict[str, float], radius_miles: float = 25) -> str:
    """
    Get the search region string for a square area around the centroid.
    """
    # Approximate miles per degree of latitude
    miles_per_lat = 69.0

    # Calculate the change in latitude for the given radius
    lat_change = radius_miles / miles_per_lat

    # Calculate the change in longitude, which varies with latitude
    lon_change = radius_miles / \
        (math.cos(math.radians(centroid['lat'])) * miles_per_lat)

    north = centroid['lat'] + lat_change
    south = centroid['lat'] - lat_change
    east = centroid['lon'] + lon_change
    west = centroid['lon'] - lon_change

    return f"{north},{east},{south},{west}"


# Example usage:
locations = [
    {"lat": 37.7749, "lon": -122.4194},  # San Francisco
    {"lat": 34.0522, "lon": -118.2437},  # Los Angeles
    {"lat": 36.1699, "lon": -115.1398}   # Las Vegas
]

centroid = calculate_centroid(locations)
print(f"Centroid: {centroid}")

search_region = get_search_region(centroid)
print(f"Search Region: {search_region}")
