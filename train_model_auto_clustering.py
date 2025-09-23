import pandas as pd
from pycaret.clustering import setup, create_model, save_model, tune_model

# 1. Wczytaj nowe dane
data = pd.read_csv("welcome_survey_simple_v2.csv", sep=";")

# 2. Konfiguracja środowiska PyCaret
setup(
    data,
    session_id=123,
    normalize=True
)

# 3. Utwórz bazowy model KMeans
kmeans = create_model("kmeans")

# 4. Dobierz najlepsze parametry (np. liczba klastrów) optymalizując silhouette
best_kmeans = tune_model(kmeans, optimize="silhouette")

# 5. Zapisz wytrenowany model
save_model(best_kmeans, "welcome_survey_auto_clustering_pipeline_v2")

print("✅ Nowy model został zapisany jako welcome_survey_auto_clustering_pipeline_v2.pkl")
