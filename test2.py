# import requests
# requests.get("https://maps-api.apple.com/v1/token", headers={
#     "alg": "ES256",
#     "kid": "65L67XZAD9",
#     "typ": "JWT",
#     "iss": "DR8M2Z7LQW",
#     "iat": 1437179036,
#     "exp": 1493298100,
#     "origin": "*.example.com"

# })

import jwt
import time
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import os
from dotenv import load_dotenv
import requests
load_dotenv()

# Replace with your actual values
team_id = os.getenv('TEAM_ID')
key_id = os.getenv('KEY_ID')
PK = os.getenv('PRIVATE_KEY')
private_key = f"""-----BEGIN PRIVATE KEY-----
{PK}
-----END PRIVATE KEY-----"""
private_key = private_key.strip()

# Load the private key to ensure it is correctly formatted
try:
    private_key_bytes = private_key.encode('utf-8')
    loaded_private_key = serialization.load_pem_private_key(
        private_key_bytes,
        password=None,
        backend=default_backend()
    )
    print("Private key loaded successfully.")
except ValueError as e:
    print("Failed to load private key:", e)
    raise

# Set the current time and expiration time
current_time = int(time.time())
expiration_time = current_time + 3600  # Token valid for 1 hour (3600 seconds)

# Create the headers
headers = {
    'alg': 'ES256',
    'kid': key_id,
    'typ': 'JWT'
}

# Create the payload
payload = {
    'iss': team_id,
    'iat': current_time,
    'exp': expiration_time
}

# Encode the token
try:
    token = jwt.encode(payload, private_key,
                       algorithm='ES256', headers=headers)
    print(f"Generated Token: {token}")
except Exception as e:
    print("Failed to generate JWT:", e)
    raise
