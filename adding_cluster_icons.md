## DeepSeek

Here's how you can add generated images/icons to represent each cluster. I'll provide you with several approaches:

## Option 1: Using AI Image Generation API (Recommended)

```python
import requests
import base64
from io import BytesIO
from PIL import Image

# Add this function to generate images
@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_cluster_image(cluster_name, description):
    """Generate an image using an AI API like DALL-E, Stable Diffusion, etc."""
    try:
        # Example using Hugging Face API (you'll need an API key)
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        headers = {"Authorization": f"Bearer {env.get('HUGGINGFACE_API_KEY')}"}
        
        prompt = f"Minimalistic icon representing: {cluster_name}. {description}. Simple, clean, professional design"
        
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        image = Image.open(BytesIO(response.content))
        return image
        
    except Exception as e:
        # Fallback to local icons if API fails
        return get_fallback_icon(cluster_name)

def get_fallback_icon(cluster_name):
    """Fallback to emoji or simple icons"""
    icon_map = {
        "explorer": "🧭",
        "thinker": "💭",
        "socializer": "👥",
        "creator": "🎨",
        "adventurer": "⚡",
        # Add more mappings based on your cluster names
    }
    
    # Find best match
    for key, icon in icon_map.items():
        if key.lower() in cluster_name.lower():
            return icon
    
    return "👤"  # Default icon

# Modify your display section:
st.header("Twoja grupa")

# Create columns for image and text
col1, col2 = st.columns([1, 3])

with col1:
    # Generate or get image
    cluster_image = generate_cluster_image(
        predicted_cluster_data['name'], 
        predicted_cluster_data['description']
    )
    
    if isinstance(cluster_image, Image.Image):
        st.image(cluster_image, width=150)
    else:
        # If it's an emoji, display it large
        st.markdown(f"<h1 style='text-align: center; font-size: 80px;'>{cluster_image}</h1>", 
                   unsafe_allow_html=True)

with col2:
    st.header(f"{predicted_cluster_data['name']}")
    st.markdown(predicted_cluster_data['description'])
```

## Option 2: Pre-generated Local Images

```python
# Add this function
def get_cluster_image(cluster_name):
    """Get pre-generated images for each cluster"""
    image_paths = {
        "Explorer": "images/explorer.png",
        "Thinker": "images/thinker.png",
        "Socializer": "images/socializer.png",
        "Creator": "images/creator.png",
        "Adventurer": "images/adventurer.png",
    }
    
    try:
        return Image.open(image_paths.get(cluster_name, "images/default.png"))
    except:
        return get_fallback_icon(cluster_name)

# Modify your display section:
st.header("Twoja grupa")

col1, col2 = st.columns([1, 3])

with col1:
    cluster_image = get_cluster_image(predicted_cluster_data['name'])
    if isinstance(cluster_image, Image.Image):
        st.image(cluster_image, width=150)
    else:
        st.markdown(f"<h1 style='text-align: center; font-size: 80px;'>{cluster_image}</h1>", 
                   unsafe_allow_html=True)

with col2:
    st.header(f"{predicted_cluster_data['name']}")
    st.markdown(predicted_cluster_data['description'])
```

## Option 3: SVG Icons with Plotly

```python
import plotly.graph_objects as go

def create_icon_figure(icon_symbol, color="#4CAF50"):
    """Create a Plotly figure with an icon"""
    fig = go.Figure(go.Scatter(
        x=[0.5], y=[0.5],
        text=[icon_symbol],
        mode="text",
        textfont=dict(size=80, color=color)
    ))
    
    fig.update_layout(
        width=150,
        height=150,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Icon mapping based on cluster characteristics
def get_cluster_icon(cluster_name, description):
    icon_map = {
        "explorer": "🧭",
        "thinker": "💡",
        "socializer": "👥",
        "creator": "🎨",
        "adventurer": "⚡",
        "analyst": "📊",
        "dreamer": "🌙",
        "leader": "👑"
    }
    
    # Find the best matching icon
    cluster_lower = cluster_name.lower()
    for key, icon in icon_map.items():
        if key in cluster_lower:
            return icon
    
    # Fallback based on description keywords
    desc_lower = description.lower()
    if any(word in desc_lower for word in ['creative', 'art', 'design']):
        return "🎨"
    elif any(word in desc_lower for word in ['social', 'people', 'team']):
        return "👥"
    elif any(word in desc_lower for word in ['think', 'analyze', 'logic']):
        return "💡"
    elif any(word in desc_lower for word in ['explore', 'travel', 'discover']):
        return "🧭"
    
    return "👤"

# Modified display section:
st.header("Twoja grupa")

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # Center the content
    icon_symbol = get_cluster_icon(
        predicted_cluster_data['name'], 
        predicted_cluster_data['description']
    )
    
    # Display large icon
    st.markdown(f"<div style='text-align: center; font-size: 80px;'>{icon_symbol}</div>", 
               unsafe_allow_html=True)
    
    st.header(f"{predicted_cluster_data['name']}")
    st.markdown(predicted_cluster_data['description'])
```

## Option 4: Advanced with Custom CSS and Colors

```python
# Add CSS styling
st.markdown("""
<style>
    .cluster-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        margin: 10px 0;
    }
    .cluster-icon {
        font-size: 80px;
        text-align: center;
        margin-bottom: 10px;
    }
    .cluster-name {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Color and icon mapping
CLUSTER_THEMES = {
    "Explorer": {"icon": "🧭", "color": "#FF6B6B"},
    "Thinker": {"icon": "💡", "color": "#4ECDC4"},
    "Socializer": {"icon": "👥", "color": "#45B7D1"},
    "Creator": {"icon": "🎨", "color": "#96CEB4"},
    "Adventurer": {"icon": "⚡", "color": "#FECA57"},
}

def get_cluster_theme(cluster_name):
    return CLUSTER_THEMES.get(cluster_name, {"icon": "👤", "color": "#95A5A6"})

# Modified display section:
theme = get_cluster_theme(predicted_cluster_data['name'])

st.markdown(f"""
<div class="cluster-card" style="background: linear-gradient(135deg, {theme['color']} 0%, {theme['color']}80 100%);">
    <div class="cluster-icon">{theme['icon']}</div>
    <div class="cluster-name">{predicted_cluster_data['name']}</div>
    <div style="text-align: center; font-size: 16px;">{predicted_cluster_data['description']}</div>
</div>
""", unsafe_allow_html=True)

st.metric("Liczba twoich znajomych", len(same_cluster_df))
```

## Recommended Approach:

I suggest starting with **Option 3** (SVG Icons with Plotly) as it's:
- Easy to implement
- No external dependencies
- Visually appealing
- Fast loading

If you want more sophisticated images, go with **Option 1** but you'll need an API key from services like:
- Hugging Face API
- OpenAI DALL-E
- Stable Diffusion API

Choose the option that best fits your technical capabilities and design requirements!