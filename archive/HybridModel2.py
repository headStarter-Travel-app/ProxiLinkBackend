
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
def load_and_preprocess_data(places_path, user_profiles_path, ratings_path):
    places_df = pd.read_csv(places_path)
    user_profiles_df = pd.read_csv(user_profiles_path)
    ratings_df = pd.read_csv(ratings_path)
    
    # Preprocess places data
    places_df['combined_category'] = places_df.apply(lambda row: [row['category'], row['category2']], axis=1)
    places_df['lat'] = places_df['location'].apply(lambda x: eval(x)['lat'])
    places_df['lon'] = places_df['location'].apply(lambda x: eval(x)['lon'])
    
    # Preprocess user profiles data
    user_profiles_df['combined_preferences'] = user_profiles_df.apply(lambda row: row['Theme'] + row['Other'], axis=1)
    
    return places_df, user_profiles_df, ratings_df

# Encode features
def encode_features(places_df, user_profiles_df):
    mlb = MultiLabelBinarizer()
    le_name = LabelEncoder()
    le_address = LabelEncoder()
    
    all_categories = list(places_df['combined_category'].explode().unique()) + list(user_profiles_df['combined_preferences'].explode().unique())
    mlb.fit([all_categories])
    
    places_encoded = mlb.transform(places_df['combined_category'])
    user_encoded = mlb.transform(user_profiles_df['combined_preferences'])
    
    places_df['name_encoded'] = le_name.fit_transform(places_df['name'])
    places_df['address_encoded'] = le_address.fit_transform(places_df['address'])
    
    places_features = np.hstack([
        places_df[['name_encoded', 'address_encoded', 'lat', 'lon']].values,
        places_encoded
    ])
    
    user_features = np.hstack([
        user_encoded,
        user_profiles_df[['Budget']].values
    ])
    
    return places_features, user_features, le_name, le_address

# Hybrid Model
class HybridModel(nn.Module):
    def __init__(self, num_users, num_places, user_features_dim, place_features_dim, embedding_dim=50):
        super(HybridModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.place_embedding = nn.Embedding(num_places, embedding_dim)
        
        self.user_features_fc = nn.Sequential(
            nn.Linear(user_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.place_features_fc = nn.Sequential(
            nn.Linear(place_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, user_ids, place_ids, user_features, place_features):
        user_emb = self.user_embedding(user_ids)
        place_emb = self.place_embedding(place_ids)
        
        user_feat = self.user_features_fc(user_features)
        place_feat = self.place_features_fc(place_features)
        
        combined = torch.cat([user_emb, place_emb, user_feat, place_feat], dim=1)
        return self.fc(combined).squeeze()

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for user_ids, place_ids, user_features, place_features, ratings in train_loader:
            optimizer.zero_grad()
            outputs = model(user_ids, place_ids, user_features, place_features)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for user_ids, place_ids, user_features, place_features, ratings in val_loader:
                outputs = model(user_ids, place_ids, user_features, place_features)
                loss = criterion(outputs, ratings)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

# Recommendation function
def get_recommendations(model, user_id, user_features, places_df, places_features, top_n=10):
    model.eval()
    with torch.no_grad():
        user_ids = torch.tensor([user_id] * len(places_df))
        place_ids = torch.tensor(range(len(places_df)))
        user_features = torch.tensor(user_features).float().repeat(len(places_df), 1)
        place_features = torch.tensor(places_features).float()
        
        scores = model(user_ids, place_ids, user_features, place_features)
        
    recommendations = places_df.copy()
    recommendations['score'] = scores.numpy()
    recommendations = recommendations.sort_values('score', ascending=False).head(top_n)
    
    return recommendations[['name', 'address', 'category', 'score']]

# Group recommendation function
def get_group_recommendations(model, user_ids, user_features_list, places_df, places_features, top_n=10, strategy='average'):
    model.eval()
    all_scores = []
    
    with torch.no_grad():
        for user_id, user_features in zip(user_ids, user_features_list):
            user_ids_tensor = torch.tensor([user_id] * len(places_df))
            place_ids = torch.tensor(range(len(places_df)))
            user_features_tensor = torch.tensor(user_features).float().repeat(len(places_df), 1)
            place_features = torch.tensor(places_features).float()
            
            scores = model(user_ids_tensor, place_ids, user_features_tensor, place_features)
            all_scores.append(scores.numpy())
    
    if strategy == 'average':
        group_scores = np.mean(all_scores, axis=0)
    elif strategy == 'least_misery':
        group_scores = np.min(all_scores, axis=0)
    else:
        raise ValueError("Invalid strategy. Choose 'average' or 'least_misery'.")
    
    recommendations = places_df.copy()
    recommendations['score'] = group_scores
    recommendations = recommendations.sort_values('score', ascending=False).head(top_n)
    
    return recommendations[['name', 'address', 'category', 'score']]

# Main function to run the hybrid model
def run_hybrid_model(places_path, user_profiles_path, ratings_path):
    # Load and preprocess data
    places_df, user_profiles_df, ratings_df = load_and_preprocess_data(places_path, user_profiles_path, ratings_path)
    
    # Encode features
    places_features, user_features, le_name, le_address = encode_features(places_df, user_profiles_df)
    
    # Prepare data for training
    ratings_df['user_id'] = ratings_df['user_id'].astype('category').cat.codes
    ratings_df['place_id'] = ratings_df['place_id'].astype('category').cat.codes
    
    # Split data
    train_data, val_data = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        list(zip(train_data['user_id'], train_data['place_id'], 
                 user_features[train_data['user_id']], places_features[train_data['place_id']], 
                 train_data['rating'])),
        batch_size=64, shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(
        list(zip(val_data['user_id'], val_data['place_id'], 
                 user_features[val_data['user_id']], places_features[val_data['place_id']], 
                 val_data['rating'])),
        batch_size=64, shuffle=False)
    
    # Initialize model
    model = HybridModel(
        num_users=len(user_profiles_df),
        num_places=len(places_df),
        user_features_dim=user_features.shape[1],
        place_features_dim=places_features.shape[1]
    )
    
    # Train model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
    
    # Example: Get recommendations for a single user
    user_id = 0
    recommendations = get_recommendations(model, user_id, user_features[user_id], places_df, places_features)
    print("Recommendations for user", user_id)
    print(recommendations)
    
    # Example: Get group recommendations
    group_user_ids = [0, 1, 2]
    group_recommendations = get_group_recommendations(model, group_user_ids, user_features[group_user_ids], places_df, places_features)
    print("\nGroup Recommendations")
    print(group_recommendations)

if __name__ == "__main__":
    places_path = "places.csv"
    user_profiles_path = "user_profiles.csv"
    ratings_path = "ratings.csv"
    run_hybrid_model(places_path, user_profiles_path, ratings_path)