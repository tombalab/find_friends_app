# --- Imports & Setup ---
import json
import streamlit as st
import pandas as pd
import plotly.express as px

# ✅ PyCaret is heavy — pin version in requirements.txt
from pycaret.clustering import load_model, predict_model  

# --- Qdrant client (optional, for external DB) ---
# Only initialize if QDRANT_URL/QDRANT_API_KEY are set in Streamlit Secrets
from qdrant_client import QdrantClient

# --- Secrets (Streamlit Cloud way) ---
env = {}
if 'QDRANT_URL' in st.secrets:
    env['QDRANT_URL'] = st.secrets['QDRANT_URL']
if 'QDRANT_API_KEY' in st.secrets:
    env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']

# Example: if you want to use Qdrant (not mandatory for CSV version)
if env.get("QDRANT_URL") and env.get("QDRANT_API_KEY"):
    qdrant = QdrantClient(
        url=env["QDRANT_URL"],
        api_key=env["QDRANT_API_KEY"]
    )

# --- Filenames (must exist in repo or in Git LFS if large) ---
MODEL_NAME = "welcome_survey_clustering_pipeline_v1"
DATA = "welcome_survey_simple_v1.csv"
CLUSTER_NAMES_AND_DESCRIPTIONS = "welcome_survey_cluster_names_and_descriptions_v1.json"
QDRANT_COLLECTION_NAME = "find_friends_app"


# --- Helper Functions with Caching ---
@st.cache_resource
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    # Use the global model inside instead of passing it
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters


# --- Sidebar: User Input ---
with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")

    age = st.selectbox("Wiek", ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender,
    }])


# --- Load resources ---
model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

# --- Prediction for this user ---
predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[str(predicted_cluster_id)]

# --- Show results ---
st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

st.header("Osoby z grupy")

# --- Visualizations ---
for col, title, xtitle in [
    ("age", "Rozkład wieku w grupie", "Wiek"),
    ("edu_level", "Rozkład wykształcenia w grupie", "Wykształcenie"),
    ("fav_animals", "Rozkład ulubionych zwierząt w grupie", "Ulubione zwierzęta"),
    ("fav_place", "Rozkład ulubionych miejsc w grupie", "Ulubione miejsce"),
    ("gender", "Rozkład płci w grupie", "Płeć"),
]:
    fig = px.histogram(same_cluster_df.sort_values(col), x=col)
    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)
