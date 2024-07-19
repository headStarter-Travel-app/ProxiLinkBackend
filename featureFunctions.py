from typing import List, Dict
from geopy.distance import great_circle

def calculate_centroid(locations: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate the centroid of given locations.
    """
    latitudes = [loc['lat'] for loc in locations]
    longitudes = [loc['lon'] for loc in locations]
    
    centroid_lat = sum(latitudes) / len(locations)
    centroid_lon = sum(longitudes) / len(locations)
    
    return {"lat": centroid_lat, "lon": centroid_lon}
