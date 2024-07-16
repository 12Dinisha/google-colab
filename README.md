# Music Recommendation by Spotify with Machine learning using Python ---- Project.....

import os
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib as mp

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('data.csv')
genre_data = pd.read_csv('data_by_genres.csv')
year_data = pd.read_csv('data_by_year.csv')

print(data.info())

print(genre_data.info())

print(year_data.info())

import pandas as pd
import numpy as np
from yellowbrick.target import FeatureCorrelation
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence','duration_ms','explicit','key','mode','year']

X, y = data[feature_names], data['popularity']

# Handle infinities and NaNs
X = np.nan_to_num(X)  # Replace NaNs with zeros and infinities with large finite values
y = np.nan_to_num(y)  # Replace NaNs with zeros and infinities with large finite values

# Create a list of the feature names
features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

plt.rcParams['figure.figsize']=(20,20)
visualizer.fit(X, y)     # Fit the data to the visualizer
visualizer.show()

def get_decade(year):
    if pd.isna(year):  # Check if the year is NaN
        return 'Unknown'  # Or any other placeholder you prefer
    period_start = int(year/10) * 10
    decade = '{}s'.format(period_start)
    return decade

data['decade'] = data['year'].apply(get_decade)

sns.set(rc={'figure.figsize':(11 ,6)})
sns.countplot(data['decade'])

sound_features = ['acousticness',



'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
fig = px.line(year_data, x='year', y=sound_features)
fig.show()


top10_genres = genre_data.nlargest(10, 'popularity')

fig = px.bar(top10_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], barmode='group')
fig.show()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))]) # Removed n_jobs parameter as it is deprecated
X = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X)
genre_data['cluster'] = cluster_pipeline.predict(X)

# Visualizing the Clusters with t-SNE

from sklearn.manifold import TSNE

tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
fig.show()

from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
kmeans.labels_
kmeans.predict([[0, 0], [12, 3]])
kmeans.cluster_centers_

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('kmeans', KMeans(n_clusters=20,
                                   verbose=False)) # Removed n_jobs parameter
                                 ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

# Visualizing the Clusters with PCA

from sklearn.decomposition import PCA
import plotly.express as px # Import the plotly.express library

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()

!pip install spotipy

%env SPOTIFY_CLIENT_ID='94f4001040da43f388cf6ef2f1c1ab27'
%env SPOTIFY_CLIENT_SECRET='01b12c28fd154a1aadcf6519e127b1f3'

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import os  # Import the os module

# Access environment variables using their names
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.environ["SPOTIFY_CLIENT_ID"],
    client_secret=os.environ["SPOTIFY_CLIENT_SECRET"]
))

# ... (Rest of your code remains the same)

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ["SPOTIFY_CLIENT_ID"],
                                                           client_secret=os.environ["SPOTIFY_CLIENT_SECRET"]))

def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):

    try:
        song_data = spotify_data[(spotify_data['name'] == song['name'])
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data

    except IndexError:
        return find_song(song['name'], song['year'])


def get_mean_vector(song_list, spotify_data):

    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):

    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []

    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    return flattened_dict


def recommend_songs( song_list, spotify_data, n_songs=10):

    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import numpy as np # Import numpy

# ... (Rest of your code)

def get_mean_vector(song_list, spotify_data):

    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        # Ensure all number_cols are present, fill missing with 0
        song_vector = song_data[number_cols].fillna(0).values
        song_vectors.append(song_vector)

    song_matrix = np.array(song_vectors)
    return np.mean(song_matrix, axis=0)

# ... (Rest of your code)

def recommend_songs( song_list, spotify_data, n_songs=10):

    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)

    # Check if 'cluster_label' is in the columns of spotify_data
    if 'cluster_label' in spotify_data.columns:
        # If present, drop it before scaling
        spotify_data_scaled = spotify_data.drop('cluster_label', axis=1)
    else:
        spotify_data_scaled = spotify_data.copy()

    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data_scaled[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data_scaled.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

def recommend_songs():
    """

    """
    song_list=[{'name': 'Raabta', 'year':2012},
                {'name': 'Phir Mohabbat', 'year': 2011}]
    # This code is incomplete and requires more context to fix.
# Please provide the surrounding code for a proper fix.

# Example:
data = []
data.append([{'name': 'Raabta', 'year':2012},
               {'name': 'Phir Mohabbat', 'year': 2011},
               {'name': 'Pal Pal', 'year': 2013},
                {'name': 'Humdard', 'year': 2014},
                {'name': 'Baatein Ye Kabhi Na', 'year': 2015}])
print(data)

[{'name': 'Sooraj Dooba Hain - From "Roy"',
  'year': 2015,
  'artists': "[' Arijit Singh']"},
 {'name': 'Sukoon Mila', 'year': 2014, 'artists': "[' Arijit Singh']"},
 {'name': 'Agar Tum Saath Ho', 'year': 2015, 'artists': "[' Arijit Singh']"},
 {'name': 'O Maahi', 'year': 2023, 'artists': "[' Arijit Singh']"},
 {'name': 'Enna Sona', 'year': 2017, 'artists': "[' Arijit Singh']"},
 {'name': 'Pal', 'year': 2018, 'artists': "[' Arijit Singh']"},
 {'name': 'Ve Kamleya', 'year': 2023, 'artists': "[' Arijit Singh']"},
 {'name': 'Woh Din',
  'year': 2019,
  'artists': "[' Arijit Singh']"},
 {'name': "Shayad", 'year': 2020, 'artists': "[' Arijit Singh']"},
 {'name': "Kesariya",
  'year': 2022,
  'artists': "[' Arijit Singh']"},
 {'name': "O Maahi",
  'year': 2023,
  'artists': "[' Arijit Singh']"},
 {'name': "Halka Halka Sa",
  'year': 2024,
  'artists': "[' Arijit Singh']"},
 {'name': "Lutt Putt Gaya",
  'year': 2023,
  'artists': "[' Arijit Singh']"}]
