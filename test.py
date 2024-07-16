import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define the dataset
class LocationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]['features']
        rating = self.data[idx]['rating']
        return torch.tensor(features, dtype=torch.float32), torch.tensor(rating, dtype=torch.float32)

# Define the model
class LocationRatingModel(nn.Module):
    def __init__(self, input_size):
        super(LocationRatingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Prepare the data
data = [
    {'features': [37.7749, -122.4194, 1, 0, 0, 0, 1, 0], 'rating': 4.5},  # Example data point
    # Add more data points as needed
]

dataset = LocationDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
input_size = len(data[0]['features'])
model = LocationRatingModel(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for features, rating in dataloader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, rating.unsqueeze(1))
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Function to get recommendations
def get_recommendations(locations, themes, interests):
    model.eval()
    # Preprocess locations, themes, interests into feature vectors
    input_features = []
    for location in locations:
        feature_vector = location + themes + interests  # Example of combining features
        input_features.append(feature_vector)
    
    input_tensor = torch.tensor(input_features, dtype=torch.float32)
    with torch.no_grad():
        scores = model(input_tensor)
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_locations = [locations[i] for i in sorted_indices]
    return sorted_locations

# Example usage
locations = [
    [37.7749, -122.4194],  # Example coordinates for San Francisco
    [34.0522, -118.2437],  # Example coordinates for Los Angeles
    # Add more locations as needed
]
themes = [1, 0, 0]  # Example themes vector
interests = [0, 1, 0]  # Example interests vector

recommendations = get_recommendations(locations, themes, interests)
print(recommendations)