# app/services/apple_maps.py

import os
from fastapi import HTTPException
import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
from typing import List, Dict

from services.appleSetup import AppleAuth

SERVER_ENDPOINT: str = "https://maps-api.apple.com"


class AppleMapsService:
    def __init__(self):
        if (os.getenv("DEV")):
            self.token = os.getenv("TOKEN_TEMP")
        else:
            self.token = AppleAuth.generate_apple_token()
        self.setup_token_update()

    def setup_token_update(self):
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            func=self.update_token,
            trigger=IntervalTrigger(hours=168),
            id='apple_token_update',
            name='Update Apple Token every 7 days',
            replace_existing=True
        )
        scheduler.start()

    def update_token(self):
        self.token = AppleAuth.generate_apple_token()
        print(f"Token updated at {datetime.now()}")

    async def get_access_token(self) -> str:
        """
        Get an access token using the auth token.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{SERVER_ENDPOINT}/v1/token",
                    headers={
                        "Authorization": f"Bearer {self.token}"
                    }
                )
                response.raise_for_status()
                token_data = response.json()
                return token_data.get("accessToken")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code,
                                detail=f"Error getting access token: {str(e)}")

    async def search(self, query: str, lat: float, lon: float) -> List[Dict]:
        try:
            access_token = await self.get_access_token()
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{SERVER_ENDPOINT}/v1/search",
                    params={
                        "q": query,
                        "searchLocation": f"{lat},{lon}",
                    },
                    headers={
                        "Authorization": f"Bearer {access_token}"
                    }
                )
                response.raise_for_status()
                results = response.json()

                locations = [
                    {
                        "name": place.get("name"),
                        "address": ", ".join(place.get("formattedAddressLines", [])),
                        "location": {
                            "lat": place.get("coordinate", {}).get("latitude"),
                            "lon": place.get("coordinate", {}).get("longitude")
                        },
                        "category": place.get("poiCategory")
                    }
                    for place in results.get("results", [])
                ]

                return locations

        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code,
                                detail=f"Apple Maps API error: {str(e)}")
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500, detail=f"Request error: {str(e)}")


apple_maps_service = AppleMapsService()
