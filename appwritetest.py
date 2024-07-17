# app.py
from dotenv import load_dotenv
import os

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import appwrite
from appwrite.client import Client
from appwrite.services.users import Users
from appwrite.services.databases import Databases
from appwrite.id import ID

import googlemaps


client = Client()

(client
 .set_endpoint('https://cloud.appwrite.io/v1')  # Your API Endpoint
 .set_project('66930c61001b090ab206')  # Your project ID
 .set_key(os.getenv('APPWRITE_API_KEY'))  # Your secret API key
 .set_self_signed()  # Use only on dev mode with a self-signed SSL cert
 )

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Initialize Appwrite client here Namann

databases = Databases(client)

result = databases.list(
)

print(result)
