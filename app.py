# The app loads a pretrained clustering model (PyCaret).
# It takes user input (age, education, favorite animals, favorite places, gender).
# The model predicts which cluster (group) the user belongs to.
# It shows:
    # The cluster’s name and description (from a JSON file).
    # The distribution of people in the same cluster (age, education, animals, places, gender).


# --- 1. Imports & Setup ---
    # streamlit for the web UI.
    # pandas for data handling.
    # pycaret.clustering to load a model and predict cluster assignments.
    # plotly.express for interactive charts.
    # JSON for loading cluster descriptions.

import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
from qdrant_client import QdrantClient
from dotenv import dotenv_values

env = dotenv_values(".env")

### Secrets using Streamlit Cloud Mechanism
# https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
# if 'QDRANT_URL' in st.secrets:
#    env['QDRANT_URL'] = st.secrets['QDRANT_URL']
# if 'QDRANT_API_KEY' in st.secrets:
#    env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']
###


MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'

DATA = 'welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

# --- 2. Helper Functions with Caching --- 
    # get_model() → loads the saved clustering model once.
    # get_cluster_names_and_descriptions() → loads cluster labels & explanations from JSON.
    # get_all_participants() → loads dataset of all survey participants, adds their predicted clusters.
# These are wrapped with @st.cache_data, so results don’t reload every time the app refreshes.

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

# --- 3. User Input Form (Sidebar) ---
    # Sidebar asks the user about themselves:
        # Age group
        # Education level
        # Favorite animals
        # Favorite place
        # Gender
    
with st.sidebar:
    st.header("Znajdujemy osoby o podobnych zainteresowaniach")
    st.markdown("Podaj swój wiek, wykształcenie, ulubione zwierzę, ulubione miejsce na wypoczynek i płeć, a my pomożemy ci znaleźć osoby o podobne do ciebie")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    # Creates a DataFrame (person_df) with that single user’s info.
    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

# --- 4. Model Prediction ---

# Loads the model and dataset of all participants.
model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

# Predicts the user’s cluster:
predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]

# Looks up name & description of that cluster from the JSON file.
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

# --- 5. Display User’s Cluster ---

# Shows cluster name + description
st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])

# Filters dataset to participants in the same cluster:
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]

# Shows how many participants belong to that cluster (st.metric)
st.metric("Liczba twoich znajomych", len(same_cluster_df))

# --- 6. Visualizations of Group ---

st.header("Osoby z grupy")

# Creates histograms with Plotly (Age distribution, Education distribution, Favorite animals, Favorite places, Gender)
# Each histogram is interactive and shows how the group is composed.
fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

