from dotenv import load_dotenv
import os
import googlemaps
import torch
import torch.nn as nn

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

# Load the trained PyTorch model
class RecommendationModel(nn.Module):
    def __init__(self):
        super(RecommendationModel, self).__init__()
        # Define your model architecture here
        self.fc = nn.Linear(10, 3)  # Example architecture

    def forward(self, x):
        return self.fc(x)

model = RecommendationModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Function to get nearby restaurants based on interests
def get_recommendations(addresses, interests):
    recommendations = []
    for address, interest in zip(addresses, interests):
        # Convert interests to tensor
        interest_tensor = torch.tensor([interest], dtype=torch.float32)
        
        # Get model predictions
        with torch.no_grad():
            predicted_interests = model(interest_tensor).numpy().tolist()
        
        # Use Google Maps API to find places based on predicted interests
        places = gmaps.places_nearby(location=address, keyword=predicted_interests, radius=5000)
        recommendations.append(places)
    
    return recommendations

# Example usage
recommendations = get_recommendations(addresses, interests)
print(recommendations)