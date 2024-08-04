#=============================================
#V2: Hybrid

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import numpy as np


data = {
    "place": ["Place1", "Place2", "Place3", "Place4"],
    "type": ["restaurant", "park", "museum", "restaurant"],
    "cuisine": ["Italian", "", "", "Chinese"],
    "location": ["downtown", "suburbs", "downtown", "midtown"],
    "price_range": ["moderate", "free", "expensive", "cheap"],
    "rating": [4.5, 4.7, 4.2, 4.0]
}

df = pd.DataFrame(data)

df

#Example DF

#Overall user/Group preferences based on the quiz and AI input
user_profile = {
    "type": "restaurant",
    "cuisine": "Italian",
    "location": "downtown",
    "price_range": "moderate"
}
user_profile


le_type = LabelEncoder()
le_cuisine = LabelEncoder()
le_location = LabelEncoder()
le_price_range = LabelEncoder()

df['type'] = le_type.fit_transform(df['type'])
df['cuisine'] = le_cuisine.fit_transform(df['cuisine'])
df['location'] = le_location.fit_transform(df['location'])
df['price_range'] = le_price_range.fit_transform(df['price_range'])

#Make everything number

df



user_profile_encoded = {
    "type": le_type.transform([user_profile['type']])[0],
    "cuisine": le_cuisine.transform([user_profile['cuisine']])[0],
    "location": le_location.transform([user_profile['location']])[0],
    "price_range": le_price_range.transform([user_profile['price_range']])[0]
}
user_profile_encoded

features = ['type', 'cuisine', 'location', 'price_range']
X = df[features].values
y = df['rating'].values

print(X)
print(y)



user_vector = torch.tensor([list(user_profile_encoded.values())], dtype=torch.float32)
user_vector

#Make model Linear, can use relu

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_dim = X.shape[1]
model = SimpleNN(input_dim)
model

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    place_vectors = torch.tensor(X, dtype=torch.float32)
    user_similarity = cosine_similarity(user_vector.numpy(), place_vectors.numpy())
    df['content_similarity'] = user_similarity[0]
df



#Colalboarative starts here
ratings = pd.DataFrame({
    "user": [1, 1, 1, 2, 2, 3, 3, 3, 3],
    "place": ["Place1", "Place2", "Place3", "Place1", "Place4", "Place1", "Place2", "Place3", "Place4"],
    "rating": [5, 4, 4, 5, 4, 3, 2, 4, 4]
})

ratings

#Split
np.random.seed(3)
msk = np.random.rand(len(ratings)) < 0.8
train = ratings[msk].copy()
val = ratings[~msk].copy()
print(train)
print(val)


# Combine train and val sets to ensure encoding dictionaries include all IDs
all_ratings = pd.concat([train, val])

user_ids = all_ratings["user"].unique()
place_ids = all_ratings["place"].unique()

user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
place_id_to_index = {place_id: index for index, place_id in enumerate(place_ids)}

# Encode user and place IDs in train and val sets so we can use it
train["user"] = train["user"].apply(lambda x: user_id_to_index[x])
train["place"] = train["place"].apply(lambda x: place_id_to_index[x])
val["user"] = val["user"].apply(lambda x: user_id_to_index[x])
val["place"] = val["place"].apply(lambda x: place_id_to_index[x])

num_users = len(user_ids)
num_places = len(place_ids)
train, val

index_to_place_id = {index: place_id for place_id, index in place_id_to_index.items()}

class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)

    def forward(self, u, v):
        u = self.user_emb(u)
        v = self.item_emb(v)
        return (u*v).sum(1)
model_cf = MF(num_users, num_places, emb_size=100)
model_cf

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_cf.parameters(), lr=0.01)
#Treaining
user_tensor = torch.tensor(train["user"].values, dtype=torch.long)
place_tensor = torch.tensor(train["place"].values, dtype=torch.long)
rating_tensor = torch.tensor(train["rating"].values, dtype=torch.float32)

epochs = 1000
for epoch in range(epochs):
    model_cf.train()
    optimizer.zero_grad()
    predictions = model_cf(user_tensor, place_tensor)
    loss = criterion(predictions, rating_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


model_cf.eval()
with torch.no_grad():
    user_index = user_id_to_index[1]  # ID 1 pred
    user_tensor = torch.tensor([user_index] * num_places, dtype=torch.long)
    place_tensor = torch.tensor(list(range(num_places)), dtype=torch.long)
    predictions = model_cf(user_tensor, place_tensor)
    preds_df_cf = pd.DataFrame({
        "place": [index_to_place_id[i] for i in range(num_places)],
        "collab_similarity": predictions.numpy()
    }).sort_values(by="collab_similarity", ascending=False)

sorted_user_predictions_cf = preds_df_cf.set_index("place")["collab_similarity"]


# Combine content-based and collaborative filtering recommendations
df['collab_similarity'] = df['place'].apply(lambda x: sorted_user_predictions_cf[x] if x in sorted_user_predictions_cf.index else 0)

#Can modify and scale this to get a better hlybrid score we will show on front end
df["hybrid_score"] = df["content_similarity"] + df["collab_similarity"]

recommendations = df.sort_values(by="hybrid_score", ascending=False)

print("Combined DataFrame with Hybrid Score:")
print(recommendations)

