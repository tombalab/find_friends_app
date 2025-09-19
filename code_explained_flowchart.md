```mermaid
flowchart TD

A["Start App"] --> B["Load pretrained model (.pkl)"]
A --> C["Load dataset (CSV)"]
A --> D["Load cluster names & descriptions (JSON)"]

subgraph Sidebar - User Input
  E["User selects: Age, Education, Fav Animals, Fav Place, Gender"]
  F["Build DataFrame with single user row"]
end

B --> G["Predict cluster for user input"]
F --> G
C --> H["Predict clusters for all participants"]
B --> H

G --> I["Find matching cluster ID"]
D --> I
I --> J["Show cluster name + description"]
I --> K["Filter dataset to same cluster participants"]

K --> L["Show metric: size of cluster"]
K --> M["Plot histograms with Plotly"]
M --> M1["Age distribution"]
M --> M2["Education distribution"]
M --> M3["Favorite animals"]
M --> M4["Favorite places"]
M --> M5["Gender distribution"]

J --> N["Display results in Streamlit UI"]
L --> N
M1 --> N
M2 --> N
M3 --> N
M4 --> N
M5 --> N
