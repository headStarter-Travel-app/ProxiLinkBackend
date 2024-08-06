#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# In[32]:


data = [
  {
    "name": "Potomac Pizza",
    "address": "7777 Baltimore Ave, College Park, MD  20740, United States",
    "location": {
      "lat": 38.9873178,
      "lon": -76.9356036
    },
    "category": "Restaurant",
    "category2": "Italian food"
  },
  {
    "name": "The Spa at The Hotel at the University of Maryland",
    "address": "7777 Baltimore Ave, FL 4, College Park, MD  20740, United States",
    "location": {
      "lat": 38.9869586,
      "lon": -76.935151
    },
    "category": "Spa",
    "category2": "Spas"
  },
  {
    "name": "The Hall CP",
    "address": "4656 Hotel Drive, College Park, MD 20742, United States",
    "location": {
      "lat": 38.9861878,
      "lon": -76.9336134
    },
    "category": "Restaurant",
    "category2": "Nightlife"
  },
  {
    "name": "Dog Haus Biergarten",
    "address": "7401 Baltimore Ave, Unit 1A, College Park, MD  20740, United States",
    "location": {
      "lat": 38.9812184,
      "lon": -76.9375926
    },
    "category": "Nightlife",
    "category2": "Nightlife"
  },
  {
    "name": "Terrapins Turf",
    "address": "4410 Knox Rd, College Park, MD  20740, United States",
    "location": {
      "lat": 38.9810151,
      "lon": -76.9384146
    },
    "category": "Nightlife",
    "category2": "Nightlife"
  },
  {
    "name": "The Cornerstone Grill & Loft",
    "address": "7325 Baltimore Ave, College Park, MD  20740, United States",
    "location": {
      "lat": 38.9806529,
      "lon": -76.9375442
    },
    "category": "Nightlife",
    "category2": "Nightlife"
  },
  {
    "name": "College Park Shopping Center",
    "address": "7370 Baltimore Ave, College Park, MD  20740, United States",
    "location": {
      "lat": 38.9806676,
      "lon": -76.9390872
    },
    "category": "Store",
    "category2": "shopping"
  },
  {
    "name": "Looney's Pub",
    "address": "8150 Baltimore Ave, College Park, MD  20740, United States",
    "location": {
      "lat": 38.9914766,
      "lon": -76.9343682
    },
    "category": "Nightlife",
    "category2": "Nightlife"
  },
  {
    "name": "SeoulSpice",
    "address": "4200 Guilford Dr, College Park, MD  20740, United States",
    "location": {
      "lat": 38.9804121,
      "lon": -76.942514
    },
    "category": "Restaurant",
    "category2": "Korean food"
  },
  {
    "name": "MeatUp Korean BBQ & Bar",
    "address": "8503 Baltimore Ave, College Park, MD  20740, United States",
    "location": {
      "lat": 38.9948577,
      "lon": -76.9319577
    },
    "category": "Restaurant",
    "category2": "Korean food"
  },
  {
    "name": "Pebbles Wellness Spa",
    "address": "8507 Baltimore Ave, College Park, MD  20740, United States",
    "location": {
      "lat": 38.9953574,
      "lon": -76.9317421
    },
    "category": "Spa",
    "category2": "Spas"
  },
  {
    "name": "Paint Branch Golf Complex",
    "address": "4690 University Blvd, College Park, MD  20740, United States",
    "location": {
      "lat": 39.0038245,
      "lon": -76.93561
    },
    "category": "Golf",
    "category2": "Golf"
  },
  {
    "name": "University of Maryland Golf Course",
    "address": "3800 Golf Course Rd, College Park, MD 20742, United States",
    "location": {
      "lat": 38.9910788,
      "lon": -76.9548726
    },
    "category": "Golf",
    "category2": "Golf"
  },
  {
    "name": "Bonchon",
    "address": "6507 America Blvd, Ste 101, Hyattsville, MD  20782, United States",
    "location": {
      "lat": 38.9686852,
      "lon": -76.9517612
    },
    "category": "Restaurant",
    "category2": "Korean food"
  },
  {
    "name": "Calvert Road Disc Golf Course",
    "address": "5201 Paint Branch Pkwy, College Park, MD 20740, United States",
    "location": {
      "lat": 38.9756263,
      "lon": -76.9167584
    },
    "category": "Golf",
    "category2": "Golf"
  },
  {
    "name": "The Shoppes At Metro Station",
    "address": "6211 Belcrest Rd, Hyattsville, MD  20782, United States",
    "location": {
      "lat": 38.9655546,
      "lon": -76.9532033
    },
    "category": "Store",
    "category2": "shopping"
  },
  {
    "name": "Riversdale House Museum",
    "address": "4811 Riverdale Rd, Riverdale, MD  20737, United States",
    "location": {
      "lat": 38.9605855,
      "lon": -76.9313344
    },
    "category": "Museum",
    "category2": "Historical Sites"
  },
  {
    "name": "National Archives archeological site",
    "address": "8601 Adelphi Rd, College Park, MD  20742, United States",
    "location": {
      "lat": 39.000308,
      "lon": -76.959666
    },
    "category": "Landmark",
    "category2": "Historical Sites"
  },
  {
    "name": "Mall at Prince George's",
    "address": "3500 East West Hwy, Hyattsville, MD 20782, United States",
    "location": {
      "lat": 38.9683307,
      "lon": -76.9574207
    },
    "category": "Store",
    "category2": "shopping"
  },
  {
    "name": "Olive Garden Italian Restaurant",
    "address": "3480 East-West Hwy, Hyattsville, MD  20782, United States",
    "location": {
      "lat": 38.967275,
      "lon": -76.9585524
    },
    "category": "Restaurant",
    "category2": "Italian food"
  },
  {
    "name": "Iron Pig Korean BBQ",
    "address": "6107 Greenbelt Rd, Berwyn Heights, MD  20740, United States",
    "location": {
      "lat": 38.9969666,
      "lon": -76.9089081
    },
    "category": "Restaurant",
    "category2": "Korean food"
  },
  {
    "name": "Essential Day Spa 2",
    "address": "5331 Baltimore Ave, Ste 106, Hyattsville, MD  20781, United States",
    "location": {
      "lat": 38.9547558,
      "lon": -76.9400233
    },
    "category": "Spa",
    "category2": "Spas"
  },
  {
    "name": "Mamma Lucia",
    "address": "4734 Cherry Hill Rd, College Park, MD  20740, United States",
    "location": {
      "lat": 39.016106,
      "lon": -76.928265
    },
    "category": "Restaurant",
    "category2": "Italian food"
  },
  {
    "name": "Beltway Plaza Mall",
    "address": "6000 Greenbelt Rd, Greenbelt, MD 20770, United States",
    "location": {
      "lat": 38.9994255,
      "lon": -76.908567
    },
    "category": "Store",
    "category2": "shopping"
  },
  {
    "name": "Three Brothers Italian Restaurant",
    "address": "6160 Greenbelt Rd, Greenbelt, MD  20770, United States",
    "location": {
      "lat": 38.9994867,
      "lon": -76.9085112
    },
    "category": "Restaurant",
    "category2": "Italian food"
  },
  {
    "name": "Hollywood Shopping Center",
    "address": "9801 Rhode Island Ave, College Park, MD  20740, United States",
    "location": {
      "lat": 39.0143196,
      "lon": -76.9196925
    },
    "category": "Store",
    "category2": "shopping"
  },
  {
    "name": "Essential Day Spa",
    "address": "10250 Baltimore Ave\nUnit D\nCollege Park, MD  20740\nUnited States",
    "location": {
      "lat": 39.022529,
      "lon": -76.9259158
    },
    "category": "Spa",
    "category2": "Spas"
  },
  {
    "name": "Go Kart Track",
    "address": "4300 Kenilworth Ave, Bladensburg, MD 20710, United States",
    "location": {
      "lat": 38.9442694,
      "lon": -76.9327497
    },
    "category": "GoKart",
    "category2": "Go Kart"
  },
  {
    "name": "Parkway Spa Asian Massage",
    "address": "5507 Landover Rd, Hyattsville, MD 20784, United States",
    "location": {
      "lat": 38.939225,
      "lon": -76.9224179
    },
    "category": "Spa",
    "category2": "Spas"
  },
  {
    "name": "Hillandale Shopping Center",
    "address": "1620 Elton Rd, Silver Spring, MD  20903, United States",
    "location": {
      "lat": 39.0213229,
      "lon": -76.9752482
    },
    "category": "Store",
    "category2": "shopping"
  },
  {
    "name": "Myong Dong",
    "address": "11114 Baltimore Ave, Beltsville, MD  20705, United States",
    "location": {
      "lat": 39.035736,
      "lon": -76.9075025
    },
    "category": "Restaurant",
    "category2": "Korean food"
  },
  {
    "name": "Greenway Shopping Center",
    "address": "7547 Greenbelt Rd, Greenbelt, MD  20770, United States",
    "location": {
      "lat": 38.9922394,
      "lon": -76.8786699
    },
    "category": "Store",
    "category2": "shopping"
  },
  {
    "name": "Da Rae Won Restaurant",
    "address": "5013 Garrett Ave, Beltsville, MD  20705, United States",
    "location": {
      "lat": 39.041095,
      "lon": -76.9064398
    },
    "category": "Restaurant",
    "category2": "Korean food"
  },
  {
    "name": "Gah Rham Korean BBQ Restaurant",
    "address": "5027 Garrett Ave, Beltsville, MD  20705, United States",
    "location": {
      "lat": 39.0410352,
      "lon": -76.9056434
    },
    "category": "Restaurant",
    "category2": "Korean food"
  },
  {
    "name": "Bryant Street NE",
    "address": "680 Rhode Island Ave NE, Washington, DC  20002, United States",
    "location": {
      "lat": 38.9220672,
      "lon": -76.9969164
    },
    "category": "Store",
    "category2": "shopping"
  },
  {
    "name": "Firepan Korean BBQ",
    "address": "962 Wayne Ave, Silver Spring, MD  20910, United States",
    "location": {
      "lat": 38.9942451,
      "lon": -77.025975
    },
    "category": "Restaurant",
    "category2": "Korean food"
  },
  {
    "name": "Langston Golf Course",
    "address": "2600 Benning Rd NE, Washington, DC  20002, United States",
    "location": {
      "lat": 38.9008777,
      "lon": -76.9674435
    },
    "category": "Golf",
    "category2": "Golf"
  },
  {
    "name": "Sligo Creek Golf Course",
    "address": "9701 Sligo Creek Pkwy, Silver Spring, MD  20901, United States",
    "location": {
      "lat": 39.0139932,
      "lon": -77.0262775
    },
    "category": "Golf",
    "category2": "Golf"
  },
  {
    "name": "Granville Moore's",
    "address": "1238 H St NE, Washington, DC  20002, United States",
    "location": {
      "lat": 38.9004267,
      "lon": -76.988931
    },
    "category": "Restaurant",
    "category2": "Belgian food"
  },
  {
    "name": "Gunpowder Golf Course",
    "address": "14300 Old Gunpowder Rd, Laurel, MD 20707, United States",
    "location": {
      "lat": 39.0864269,
      "lon": -76.9201451
    },
    "category": "Golf",
    "category2": "Golf"
  },
  {
    "name": "Rock Creek Park Golf Course",
    "address": "6100 16th St NW, Washington, DC  20011, United States",
    "location": {
      "lat": 38.966433,
      "lon": -77.0407677
    },
    "category": "Golf",
    "category2": "Golf"
  },
  {
    "name": "Iron Age Korean Steak House",
    "address": "3365 14th St NW, Washington, DC  20010, United States",
    "location": {
      "lat": 38.9311633,
      "lon": -77.032367
    },
    "category": "Restaurant",
    "category2": "Korean food"
  },
  {
    "name": "9:30 Club",
    "address": "815 V St NW, Washington, DC  20001, United States",
    "location": {
      "lat": 38.9178869,
      "lon": -77.0237142
    },
    "category": "MusicVenue",
    "category2": "Nightlife"
  },
  {
    "name": "Morgan Black Spas",
    "address": "10111 Martin Luther King Jr Hwy, Bowie, MD  20720, United States",
    "location": {
      "lat": 38.9553445,
      "lon": -76.8287839
    },
    "category": "Spa",
    "category2": "Spas"
  },
  {
    "name": "Caruso's Grocery",
    "address": "914 14th St SE, Washington, DC  20003, United States",
    "location": {
      "lat": 38.879753,
      "lon": -76.985214
    },
    "category": "Restaurant",
    "category2": "Italian food"
  },
  {
    "name": "The Little Gay Pub",
    "address": "1100 P St NW, Washington, DC 20001, United States",
    "location": {
      "lat": 38.9095139,
      "lon": -77.0273898
    },
    "category": "Nightlife",
    "category2": "Nightlife"
  },
  {
    "name": "Trattoria Alberto's Italian Cuisine",
    "address": "506 Eighth St SE, Washington, DC  20003, United States",
    "location": {
      "lat": 38.8824609,
      "lon": -76.994699
    },
    "category": "Restaurant",
    "category2": "Italian food"
  },
  {
    "name": "RPM Italian",
    "address": "650 K Street NW, Washington, DC 20001, United States",
    "location": {
      "lat": 38.9021968,
      "lon": -77.0209301
    },
    "category": "Restaurant",
    "category2": "Italian food"
  },
  {
    "name": "Belga Cafe",
    "address": "514 Eighth St SE, Washington, DC  20003, United States",
    "location": {
      "lat": 38.8821764,
      "lon": -76.9947356
    },
    "category": "Restaurant",
    "category2": "Belgian food"
  },
  {
    "name": "United States Capitol",
    "address": "Washington, DC  20001, United States",
    "location": {
      "lat": 38.8894251,
      "lon": -77.008616
    },
    "category": "Landmark",
    "category2": "Historical Sites"
  },
  {
    "name": "CityCenterDC",
    "address": "825 Tenth St NW, Washington, DC 20001, United States",
    "location": {
      "lat": 38.9004601,
      "lon": -77.0255113
    },
    "category": "Store",
    "category2": "shopping"
  },
  {
    "name": "Etalon Day Spa",
    "address": "707 D St NW, Washington, DC  20004, United States",
    "location": {
      "lat": 38.8949455,
      "lon": -77.0223134
    },
    "category": "Spa",
    "category2": "Spas"
  },
  {
    "name": "Anju",
    "address": "1805 18th St NW, Washington, DC  20009, United States",
    "location": {
      "lat": 38.9143441,
      "lon": -77.0414731
    },
    "category": "Restaurant",
    "category2": "Korean food"
  },
  {
    "name": "Aura spa - Yards",
    "address": "1212 Fourth St SE, Ste 170, Washington, DC  20003, United States",
    "location": {
      "lat": 38.8753353,
      "lon": -77.0000881
    },
    "category": "Spa",
    "category2": "Spas"
  },
  {
    "name": "Grazie Nonna",
    "address": "1100 15th St NW, Washington, DC  20005, United States",
    "location": {
      "lat": 38.9042968,
      "lon": -77.0352845
    },
    "category": "Restaurant",
    "category2": "Italian food"
  },
  {
    "name": "Spa Logic",
    "address": "1721 Connecticut Ave NW, Washington, DC  20009, United States",
    "location": {
      "lat": 38.913265,
      "lon": -77.045313
    },
    "category": "Beauty",
    "category2": "Spas"
  },
  {
    "name": "Decades",
    "address": "1219 Connecticut Avenue NW, Washington, DC 20036, United States",
    "location": {
      "lat": 38.906513,
      "lon": -77.04126
    },
    "category": "Nightlife",
    "category2": "Nightlife"
  },
  {
    "name": "Swingers Dupont Circle",
    "address": "1330 19th St NW, Washington, DC  20036, United States",
    "location": {
      "lat": 38.9084408,
      "lon": -77.0437496
    },
    "category": "MiniGolf",
    "category2": "Nightlife"
  },
  {
    "name": "St. Arnold's Mussel Bar",
    "address": "1827 Jefferson Pl NW, Washington, DC  20036, United States",
    "location": {
      "lat": 38.9065335,
      "lon": -77.0427796
    },
    "category": "Restaurant",
    "category2": "Belgian food"
  },
  {
    "name": "The White House",
    "address": "1600 Pennsylvania Ave NW, Washington, DC 20500-0003, United States",
    "location": {
      "lat": 38.8976817,
      "lon": -77.036588
    },
    "category": "Landmark",
    "category2": "Historical Sites"
  },
  {
    "name": "Enterprise Golf Course",
    "address": "2802 Enterprise Rd, Mitchellville, MD 20721, United States",
    "location": {
      "lat": 38.928356,
      "lon": -76.81691
    },
    "category": "Golf",
    "category2": "Golf"
  },
  {
    "name": "Live-K",
    "address": "730 Maine Ave SW, Washington, DC  20024, United States",
    "location": {
      "lat": 38.8788022,
      "lon": -77.0234163
    },
    "category": "Nightlife",
    "category2": "Nightlife"
  },
  {
    "name": "Washington Monument",
    "address": "2 15th St NW, Washington, DC  20024, United States",
    "location": {
      "lat": 38.88943,
      "lon": -77.0353955
    },
    "category": "Landmark",
    "category2": "Historical Sites"
  },
  {
    "name": "World War II Memorial",
    "address": "1750 Independence Ave SW, Washington, DC 20024, United States",
    "location": {
      "lat": 38.8893877,
      "lon": -77.0405209
    },
    "category": "Landmark",
    "category2": "Historical Sites"
  },
  {
    "name": "East Potomac Golf Course",
    "address": "972 Ohio Dr SW, Washington, DC  20024, United States",
    "location": {
      "lat": 38.8745842,
      "lon": -77.0267934
    },
    "category": "Golf",
    "category2": "Golf"
  },
  {
    "name": "Thomas Jefferson Memorial",
    "address": "16 E Basin Dr SW, Washington, DC 20024, United States",
    "location": {
      "lat": 38.8813621,
      "lon": -77.0365351
    },
    "category": "Landmark",
    "category2": "Historical Sites"
  },
  {
    "name": "Vietnam Veterans Memorial",
    "address": "5 Henry Bacon Dr NW, Washington, DC 20002, United States",
    "location": {
      "lat": 38.8912541,
      "lon": -77.0477146
    },
    "category": "Landmark",
    "category2": "Historical Sites"
  },
  {
    "name": "Martin Luther King, Jr. Memorial",
    "address": "1850 W Basin Dr SW, Washington, DC  20024, United States",
    "location": {
      "lat": 38.8862185,
      "lon": -77.0442197
    },
    "category": "Landmark",
    "category2": "Historical Sites"
  },
  {
    "name": "il Canale",
    "address": "1065 31st St NW, Washington, DC  20007, United States",
    "location": {
      "lat": 38.9045021,
      "lon": -77.0609403
    },
    "category": "Restaurant",
    "category2": "Italian food"
  },
  {
    "name": "Lincoln Memorial",
    "address": "2 Lincoln Memorial Cir NW, Washington, DC  20037, United States",
    "location": {
      "lat": 38.889218,
      "lon": -77.050178
    },
    "category": "Landmark",
    "category2": "Historical Sites"
  },
  {
    "name": "The Sovereign",
    "address": "1206 Wisconsin Ave NW, Washington, DC  20007, United States",
    "location": {
      "lat": 38.9055479,
      "lon": -77.063183
    },
    "category": "Restaurant",
    "category2": "Belgian food"
  },
  {
    "name": "Filomena Ristorante",
    "address": "1063 Wisconsin Ave NW, Washington, DC  20007, United States",
    "location": {
      "lat": 38.9044418,
      "lon": -77.0626614
    },
    "category": "Restaurant",
    "category2": "Italian food"
  },
  {
    "name": "Bowie Golf Club",
    "address": "7420 Laurel Bowie Rd\nBowie, MD  20715\nUnited States",
    "location": {
      "lat": 38.9960736,
      "lon": -76.7622685
    },
    "category": "Golf",
    "category2": "Golf"
  },
  {
    "name": "Et Voila",
    "address": "5120 MacArthur Blvd NW, Washington, DC  20016, United States",
    "location": {
      "lat": 38.925407,
      "lon": -77.102364
    },
    "category": "Restaurant",
    "category2": "Belgian food"
  },
  {
    "name": "Woodhouse Spa",
    "address": "2 Paseo Dr, Rockville, MD  20852, United States",
    "location": {
      "lat": 39.0426612,
      "lon": -77.112211
    },
    "category": "Spa",
    "category2": "Spas"
  },
  {
    "name": "Bowie Town Center",
    "address": "15606 Emerald Way, Bowie, MD  20716, United States",
    "location": {
      "lat": 38.9442596,
      "lon": -76.7347193
    },
    "category": "Store",
    "category2": "shopping"
  },
  {
    "name": "Mannequin Pis",
    "address": "18064 Georgia Ave, Olney, MD  20832, United States",
    "location": {
      "lat": 39.1524126,
      "lon": -77.0676249
    },
    "category": "Restaurant",
    "category2": "Belgian food"
  },
  {
    "name": "Autobahn Indoor Speedway",
    "address": "8251 Preston Ct, Ste F, Jessup, MD  20794, United States",
    "location": {
      "lat": 39.1497871,
      "lon": -76.7937002
    },
    "category": "GoKart",
    "category2": "Go Kart"
  },
  {
    "name": "K1 Speed",
    "address": "8251 Preston Ct, Jessup, MD  20794, United States",
    "location": {
      "lat": 39.1497871,
      "lon": -76.7937002
    },
    "category": "GoKart",
    "category2": "Go Kart"
  },
  {
    "name": "Crofton Go-Kart Raceway",
    "address": "1050 State Route 3 South, Gambrills, MD 21054, United States",
    "location": {
      "lat": 39.0282504,
      "lon": -76.6891592
    },
    "category": "GoKart",
    "category2": "Go Kart"
  },
  {
    "name": "United Karting",
    "address": "7206 Ridge Rd, Hanover, MD  21076, United States",
    "location": {
      "lat": 39.1798945,
      "lon": -76.7105019
    },
    "category": "GoKart",
    "category2": "Go Kart"
  },
  {
    "name": "Freetjes",
    "address": "1448 Light St, Baltimore, MD  21230, United States",
    "location": {
      "lat": 39.2729366,
      "lon": -76.6120387
    },
    "category": "Restaurant",
    "category2": "Belgian food"
  },
  {
    "name": "Cock and Bowl",
    "address": "302 Poplar Alley, Occoquan, VA 22125, United States",
    "location": {
      "lat": 38.6838156,
      "lon": -77.2610078
    },
    "category": "Restaurant",
    "category2": "Belgian food"
  },
  {
    "name": "The Brewer's Art",
    "address": "1106 N Charles St, Baltimore, MD  21201, United States",
    "location": {
      "lat": 39.3027715,
      "lon": -76.6163875
    },
    "category": "Brewery",
    "category2": "Belgian food"
  },
  {
    "name": "Autobahn Indoor Speedway & Events",
    "address": "45448 E Severn Way, Unit 150, Sterling, VA 20166, United States",
    "location": {
      "lat": 39.0214199,
      "lon": -77.427635
    },
    "category": "GoKart",
    "category2": "Go Kart"
  },
  {
    "name": "K1 Speed",
    "address": "45448 E Severn Way, Sterling, VA  20166, United States",
    "location": {
      "lat": 39.0214199,
      "lon": -77.427635
    },
    "category": "GoKart",
    "category2": "Go Kart"
  },
  {
    "name": "Autobahn Indoor Speedway & Events",
    "address": "8415 Kelso Dr, Essex, MD 21221, United States",
    "location": {
      "lat": 39.326625,
      "lon": -76.4848801
    },
    "category": "GoKart",
    "category2": "Go Kart"
  },
  {
    "name": "K1 Speed",
    "address": "8300 Sudley Rd, Unit A5, Manassas, VA  20109, United States",
    "location": {
      "lat": 38.7720027,
      "lon": -77.5036097
    },
    "category": "GoKart",
    "category2": "Go Kart"
  },
  {
    "name": "Go Kart Track",
    "address": "10907 Pulaski Hwy, White Marsh, MD  21162, United States",
    "location": {
      "lat": 39.3819961,
      "lon": -76.4245172
    },
    "category": "GoKart",
    "category2": "Go Kart"
  },
  {
    "name": "Aquatic Sports",
    "address": "10803 SW Barbur Blvd, Portland, OR  97219, United States",
    "location": {
      "lat": 45.4473683,
      "lon": -122.7296765
    },
    "category": "Aquatic Sports",
    "category2": "Aquatic Sports"
  }
]

places_df = pd.DataFrame(data)
places_df
original_data = places_df.copy()
original_data


# In[33]:


#Themes
# Themes with relevant categories

romantic_date = [
    "Cafe", "Restaurant", "Bakery", "AmusementPark", "Beach", "Winery", 
    "Theater", "MovieTheater", "Park", "Zoo", "Aquarium", "Store", 
    "MiniGolf", "Bowling", "MusicVenue", "Store", "Mall"
]

family_outing = [
    "AmusementPark", "Zoo", "Aquarium", "Park", "Playground", "MovieTheater", 
    "Museum", "NationalPark", "Beach", "Campground", "FoodMarket"
]

outdoor_adventure = [
    "NationalPark", "Park", "Beach", "Hiking", "Kayaking", "Fishing", 
    "Golf", "MiniGolf", "RockClimbing", "RVPark", "SkatePark", "Skating", 
    "Skiing", "Surfing", "Swimming", "Tennis", "Volleyball"
]

educational_trip = [
    "Museum", "Library", "Aquarium", "NationalPark", "Planetarium", "Zoo", 
    "University", "Landmark", "NationalMonument", "ReligiousSite"
]

night_out = [
    "Nightlife", "Brewery", "Restaurant", "MovieTheater", "Theater", 
    "MusicVenue", "Casino", "Bar", "Store", "Winery"
]

relaxation_and_wellness = [
    "Beach", "Spa", "FitnessCenter", "Park", "Yoga", "MeditationCenter"
]

sports_and_fitness = [
    "Stadium", "FitnessCenter", "Golf", "Tennis", "Basketball", "Soccer", 
    "Baseball", "Swimming", "Volleyball", "Bowling", "RockClimbing", 
    "Hiking", "Kayaking", "Surfing", "Skating", "Skiing", "SkatePark"
]

shopping_spree = [
    "Store", "FoodMarket", "Mall", "Pharmacy"
]

kids_fun_day = [
    "AmusementPark", "Zoo", "Aquarium", "Park", "Playground", "MovieTheater", 
    "MiniGolf", "Bowling", "Fairground", "GoKart"
]

historical_and_cultural_exploration = [
    "Museum", "Castle", "Fortress", "Landmark", "NationalMonument", 
    "ReligiousSite", "Planetarium", "Fairground", "ConventionCenter"
]

Vacation = [
    "Hotel", "Beach", "NationalPark", "Park", "Winery", "Campground", 
    "Marina", "Skiing", "RVPark", "Store"
]

food_and_drink = [
    "Restaurant", "Cafe", "Bakery", "Brewery", "Winery", "FoodMarket"
]



# In[34]:


user_profile = {
    "GroupName": "Group1",
    "Theme": romantic_date,
    "Budget": 100,
    "Other": []
}
df_user_profile = pd.DataFrame([user_profile])

# Print the DataFrame
df_user_profile


#Add in Budget, Time of Day, and based on theme, categories will change (So it iwll emphasize EX on a date, Parks, Shopping, Cafe and Restaurants)
#User profile doesnt need any of the expanded shopping and stuff cuz that was already used to get all the recommendations. The Profile will have parameters form the model ONLY


# In[35]:


places_df['combined_category'] = places_df.apply(lambda row: [row['category'], row['category2']], axis=1)

# Combine Theme and Other in user_profile_df
df_user_profile['combined_preferences'] = df_user_profile.apply(lambda row: row['Theme'] + row['Other'], axis=1)
df_user_profile


# In[36]:


from sklearn.preprocessing import MultiLabelBinarizer
# Initialize encoders
# Initialize encoders
# Initialize encoders
mlb = MultiLabelBinarizer()
le_name = LabelEncoder()
le_address = LabelEncoder()

# Fit the MultiLabelBinarizer on both places and user preferences
all_categories = list(places_df['combined_category'].explode().unique()) + list(df_user_profile['combined_preferences'].explode().unique())
mlb.fit([all_categories])

# Transform the combined categories
places_encoded = mlb.transform(places_df['combined_category'])
user_encoded = mlb.transform(df_user_profile['combined_preferences'])

# Encode the name and address columns
places_df['name_encoded'] = le_name.fit_transform(places_df['name'])
places_df['address_encoded'] = le_address.fit_transform(places_df['address'])

# Extract latitude and longitude
places_df['lat'] = places_df['location'].apply(lambda x: x['lat'])
places_df['lon'] = places_df['location'].apply(lambda x: x['lon'])

# Create the final places feature matrix
places_features = np.hstack([places_df[['name_encoded', 'address_encoded', 'lat', 'lon']].values, places_encoded])
user_features = np.hstack([user_encoded, df_user_profile[['Budget']].values])

places_features_df = pd.DataFrame(places_features)
user_features_df = pd.DataFrame(user_features)

print("Encoded Places DataFrame:")
print(places_features_df)
print("\nEncoded User DataFrame:")
print(user_features_df)


# In[37]:


# Convert features to tensors
places_tensor = torch.tensor(places_features, dtype=torch.float32)
user_tensor = torch.tensor(user_features, dtype=torch.float32)

places_tensor.shape, user_tensor.shape


# In[38]:


places_tensor.shape, user_tensor.shape


# In[39]:


class ContentModel(nn.Module):
    def __init__(self, input_dim):
        super(ContentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc4(x)
        return x

input_dim = places_tensor.shape[1] + user_tensor.shape[1]
model = ContentModel(input_dim)

# Repeat user tensor and concatenate with place tensor
user_tensor_repeated = user_tensor.repeat(places_tensor.shape[0], 1)
user_place_tensor = torch.cat((user_tensor_repeated, places_tensor), dim=1)

# Calculate similarity (cosine similarity)
def cosine_similarity(tensor1, tensor2):
    dot_product = torch.sum(tensor1 * tensor2, dim=1)
    norm1 = torch.norm(tensor1, dim=1)
    norm2 = torch.norm(tensor2, dim=1)
    return dot_product / (norm1 * norm2)

user_pref_tensor = torch.tensor(user_encoded, dtype=torch.float32)
place_pref_tensor = torch.tensor(places_encoded, dtype=torch.float32)
similarity = cosine_similarity(user_pref_tensor, place_pref_tensor).reshape(-1, 1)

# Adjust target tensor based on similarity
target_tensor = similarity

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(user_place_tensor)
    loss = criterion(output, target_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')


# In[40]:


# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(user_place_tensor)
    predictions = predictions.numpy().flatten()

# Combine predictions with places for better understanding
recommendations = places_df.copy()
recommendations['score'] = predictions

# Decode the name and address columns
recommendations['name'] = le_name.inverse_transform(recommendations['name_encoded'])
recommendations['address'] = le_address.inverse_transform(recommendations['address_encoded'])

# Sort recommendations by score
recommendations = recommendations.sort_values(by='score', ascending=False)

print("\nTop Recommendations:")
recommendations

