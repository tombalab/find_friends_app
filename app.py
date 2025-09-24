import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
from qdrant_client import QdrantClient
from dotenv import dotenv_values

# --- Konfiguracja ---
env = dotenv_values(".env")

if 'QDRANT_URL' in st.secrets:
    env['QDRANT_URL'] = st.secrets['QDRANT_URL']
if 'QDRANT_API_KEY' in st.secrets:
    env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']

MODEL_NAME = 'welcome_survey_clustering_pipeline_v1'
DATA = 'welcome_survey_simple_v1.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v1.json'

# --- Funkcje pomocnicze ---
@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants(model):
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters

# --- Formularz boczny ---
with st.sidebar:
    st.header("🔎 Znajdź osoby o podobnych zainteresowaniach")
    st.markdown("Podaj dane, a my pokażemy ci grupę podobnych osób:")
    age = st.selectbox("Wiek", ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

# --- Model & Dane ---
model = get_model()
all_df = get_all_participants(model)
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

# --- Wyświetlenie grupy użytkownika ---

st.header(f"🎯 Najbliżej Ci do grupy: {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])

# --- Dodanie grafiki do grupy ---
# Zakładamy że obrazy są w katalogu ./images/cluster_<id>.png
image_path = f"images/cluster_{predicted_cluster_id}.png"
try:
    st.image(image_path, use_column_width=True)
except Exception:
    st.info("Brak dedykowanej grafiki dla tej grupy.")

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("👥 Liczba osób w tej grupie", len(same_cluster_df))

# --- Wizualizacje grupy ---
st.subheader("📊 Jak wygląda Twoja grupa?")

# Wiek jako bar chart
fig = px.bar(same_cluster_df['age'].value_counts().reset_index(),
             x='index', y='age',
             labels={'index': 'Wiek', 'age': 'Liczba osób'})
fig.update_layout(title="Rozkład wieku w grupie")
st.plotly_chart(fig, use_container_width=True)

# Wykształcenie - pie chart
fig = px.pie(same_cluster_df, names="edu_level", title="Wykształcenie w grupie", hole=0.4)
st.plotly_chart(fig, use_container_width=True)

# Ulubione zwierzęta - pie chart
fig = px.pie(same_cluster_df, names="fav_animals", title="Ulubione zwierzęta w grupie", hole=0.4)
st.plotly_chart(fig, use_container_width=True)

# Ulubione miejsca - pie chart
fig = px.pie(same_cluster_df, names="fav_place", title="Ulubione miejsca w grupie", hole=0.4)
st.plotly_chart(fig, use_container_width=True)

# Płeć - pie chart
fig = px.pie(same_cluster_df, names="gender", title="Rozkład płci w grupie", hole=0.4)
st.plotly_chart(fig, use_container_width=True)

# --- Porównanie użytkownika do grupy ---
st.subheader("📌 Jak wypadasz na tle grupy?")
col1, col2 = st.columns(2)
with col1:
    st.metric("Twój wiek", age)
    st.metric("Twoje wykształcenie", edu_level)
with col2:
    st.metric("Twoje ulubione zwierzęta", fav_animals)
    st.metric("Twoje ulubione miejsce", fav_place)
