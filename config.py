# app/config.py

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Proxi Link AI API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "API for Proxi Link App"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "8000"))

    # API Endpoints

    # Database Settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")

    # Appwrite Settings
    APPWRITE_ENDPOINT: str = "https://cloud.appwrite.io/v1"
    APPWRITE_PROJECT_ID: str = "66930c61001b090ab206"
    APPWRITE_API_KEY: str = os.getenv("APPWRITE_API_KEY")
    APPWRITE_DATABASE_ID: str = "66930e1000087eb0d4bd"
    APPWRITE_USER_COLLECTION_ID: str = "66930e5900107bc194dc"
    APPWRITE_PREFERENCES_COLLECTION_ID: str = "6696016b00117bbf6352"

    # Google Maps Settings
    GOOGLE_MAPS_API_KEY: str = os.getenv("GOOGLE_MAPS_API")

    # Apple Maps Settings
    APPLE_MAPS_TOKEN_UPDATE_INTERVAL: int = 168  # hours

    # Recommendation Settings
    DEFAULT_SEARCH_RADIUS: int = 5000  # meters
    MAX_RECOMMENDATIONS: int = 20

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create a global settings object
settings = Settings()

# Appwrite configuration dictionary
appwrite_config = {
    "database_id": settings.APPWRITE_DATABASE_ID,
    "user_collection_id": settings.APPWRITE_USER_COLLECTION_ID,
    "preferences_collection_id": settings.APPWRITE_PREFERENCES_COLLECTION_ID
}

# List of supported categories
SUPPORTED_CATEGORIES = [
    "Airport", "AirportGate", "AirportTerminal", "AmusementPark", "ATM",
    "Aquarium", "Bakery", "Bank", "Beach", "Brewery", "Cafe", "Campground",
    "CarRental", "EVCharger", "FireStation", "FitnessCenter", "FoodMarket",
    "GasStation", "Hospital", "Hotel", "Laundry", "Library", "Marina",
    "MovieTheater", "Museum", "NationalPark", "Nightlife", "Park", "Parking",
    "Pharmacy", "Playground", "Police", "PostOffice", "PublicTransport",
    "ReligiousSite", "Restaurant", "Restroom", "School", "Stadium", "Store",
    "Theater", "University", "Winery", "Zoo"
]
