import pandas as pd
from pycaret.clustering import setup, create_model, save_model

print("➡️ Startuję trenowanie modelu...")

# 1. Wczytaj dane
data = pd.read_csv("welcome_survey_simple_v2.csv", sep=";")
print("✅ Dane wczytane:", data.shape)

# 2. Konfiguracja PyCaret (UWAGA: w 3.3.2 nie ma silent=True)
s = setup(
    data=data,
    session_id=123,
    normalize=True,
    verbose=True
)

print("✅ Setup zakończony")

# 3. Stwórz model (np. kmeans z 5 klastrami)
kmeans = create_model("kmeans", num_clusters=5)
print("✅ Model utworzony")

# 4. Zapisz model do pliku .pkl
save_model(kmeans, "welcome_survey_clustering_pipeline_v2")
print("✅ Model zapisany jako welcome_survey_clustering_pipeline_v2.pkl")
