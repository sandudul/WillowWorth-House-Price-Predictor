import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Cache data loading to avoid re-reading file every interaction
@st.cache_data
def load_data(path="housing.csv"):
    df = pd.read_csv(path)
    df = df.dropna()  # simple cleaning as in the notebook
    return df


def set_background(image_path: str):
    """Set a background image for the Streamlit app by embedding the image as a base64 data URI and
    injecting CSS into the page. Mime type is inferred from the file extension."""
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        # infer mime type from extension
        if image_path.lower().endswith('.png'):
            mime = 'image/png'
        elif image_path.lower().endswith('.gif'):
            mime = 'image/gif'
        elif image_path.lower().endswith('.webp'):
            mime = 'image/webp'
        else:
            mime = 'image/jpeg'
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:{mime};base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not set background image: {e}")


def extract_palette(image_path: str, n_colors: int = 5):
    """Extract a palette of n_colors hex colors from the given image using KMeans clustering.
    Returns a list of hex color strings (e.g. ['#aabbcc', ...]). Falls back to a default palette on error."""
    try:
        img = Image.open(image_path).convert("RGB")
        # resize to speed up clustering
        img = img.resize((200, 200))
        arr = np.array(img).reshape(-1, 3)

        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        labels = kmeans.fit_predict(arr)
        centers = kmeans.cluster_centers_.astype(int)

        # Sort colors by cluster size (most frequent first)
        counts = np.bincount(labels)
        order = np.argsort(-counts)
        colors = [centers[i] for i in order]

        def rgb_to_hex(c):
            return '#%02x%02x%02x' % (c[0], c[1], c[2])

        return [rgb_to_hex(c) for c in colors]
    except Exception:
        # fallback palette
        return ['#2b6cb0', '#90cdf4', '#f6ad55', '#2f855a', '#ffffff']


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def luminance(hex_color: str):
    r, g, b = hex_to_rgb(hex_color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def apply_palette_css(palette):
    """Inject CSS using the extracted palette: sets primary/secondary colors, panel background and button styles."""
    primary = palette[0]
    secondary = palette[1] if len(palette) > 1 else palette[0]
    # decide text color based on primary brightness
    text_on_primary = '#000000' if luminance(primary) > 160 else '#ffffff'
    # translucent panel background using primary with alpha
    r, g, b = hex_to_rgb(primary)
    panel_rgba = f'rgba({r},{g},{b},0.08)'

    css = f"""
    <style>
        :root {{
            --primary: {primary};
            --secondary: {secondary};
            --panel-bg: {panel_rgba};
            --primary-text: #cc5500; /* burnt orange for big headings */
            --text-color: #111111; /* darker global text color */
            --big-text-color: #cc5500; /* burnt orange for big headings */
        }}

    /* Global text color (darker) - applied before headings so headings keep primary */
    .stApp, .stApp .main, .stApp p, .stApp span, .stApp label, .stApp td, .stApp th, .stApp li, .stApp a {{
        color: var(--text-color) !important;
    }}

    /* Main content panel: translucent so background shows through */
    .stApp .main .block-container {{
        background: var(--panel-bg);
        padding: 1.2rem 1.6rem;
        border-radius: 12px;
    }}

    /* Headings use burnt orange for prominence */
    .stApp h1, .stApp h2, .stApp h3 {{
        color: var(--big-text-color) !important;
        font-weight: 700 !important;
    }}

    /* Top navigation bar / header styling */
    .stApp > header, header, .css-18e3th9, .css-1v3fvcr, .css-1avcm0n {{
        background-color: var(--primary) !important;
        color: var(--primary-text) !important;
    }}
    .stApp > header a, header a, .stApp > header button {{
        color: var(--primary-text) !important;
    }}

    /* App name in top-left corner of the header */
    .stApp > header::before, header::before, .css-18e3th9::before {{
        content: "House Price Predictor";
        position: absolute;
        left: 16px;
        top: 8px;
        font-weight: 1000;
        color: var(--primary-text);
        font-size: 16px;
        z-index: 9999;
    }}

    /* Primary buttons */
    .stButton>button, .stDownloadButton>button {{
        background-color: var(--primary) !important;
        color: var(--primary-text) !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 8px !important;
        padding: 0.45rem 0.75rem !important;
    }}

    /* Number inputs / select boxes: subtle border using secondary */
    .stNumberInput, .stSelectbox, .stTextInput {{
        border-radius: 8px;
    }}

    /* Make tables and dataframes slightly translucent */
    .stApp .element-container .stDataFrame {{
        background: rgba(255,255,255,0.6);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Cache model training so it doesn't retrain on every interaction
@st.cache_resource
def train_models(df):
    # Features expected in the dataset
    feature_cols = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]

    X = df[feature_cols].copy()
    y = df["median_house_value"].copy()

    # Apply same log transforms used in notebook
    X["total_rooms"] = np.log(X["total_rooms"] + 1)
    X["total_bedrooms"] = np.log(X["total_bedrooms"] + 1)
    X["population"] = np.log(X["population"] + 1)
    X["households"] = np.log(X["households"] + 1)

    # One-hot encode ocean_proximity and join
    dummies = pd.get_dummies(df["ocean_proximity"], prefix="ocean")
    X = X.join(dummies)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaler
    scaler = StandardScaler()
    # Fit only on numeric columns (exclude one-hot columns)
    numeric_cols = [c for c in X_train.columns if not c.startswith("ocean_")]
    X_train_num = scaler.fit_transform(X_train[numeric_cols])

    # Reconstruct scaled X_train with dummy columns appended
    X_train_s = pd.DataFrame(X_train_num, columns=numeric_cols, index=X_train.index)
    X_train_s = X_train_s.join(X_train[[c for c in X_train.columns if c.startswith("ocean_")]])

    # Same for test
    X_test_num = scaler.transform(X_test[numeric_cols])
    X_test_s = pd.DataFrame(X_test_num, columns=numeric_cols, index=X_test.index)
    X_test_s = X_test_s.join(X_test[[c for c in X_test.columns if c.startswith("ocean_")]])

    # Train models
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)

    # Evaluate
    lr_score = r2_score(y_test, lr.predict(X_test_s))
    rf_score = r2_score(y_test, rf.predict(X_test_s))

    # Save metadata needed for later preprocessing/prediction
    model_metadata = {
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "dummy_cols": [c for c in X_train.columns if c.startswith("ocean_")],
        "lr": lr,
        "rf": rf,
        "lr_score": lr_score,
        "rf_score": rf_score,
    }
    return model_metadata


def preprocess_input(input_df, metadata):
    # Apply log transforms
    df = input_df.copy()
    df["total_rooms"] = np.log(df["total_rooms"] + 1)
    df["total_bedrooms"] = np.log(df["total_bedrooms"] + 1)
    df["population"] = np.log(df["population"] + 1)
    df["households"] = np.log(df["households"] + 1)

    # One-hot for ocean_proximity
    ocean = pd.get_dummies(df["ocean_proximity"], prefix="ocean")
    df = df.drop(columns=["ocean_proximity"]) 
    df = df.join(ocean)

    # Ensure all dummy cols exist
    for c in metadata["dummy_cols"]:
        if c not in df.columns:
            df[c] = 0

    # Keep columns in the same order: numeric_cols + dummy_cols
    final_cols = metadata["numeric_cols"] + metadata["dummy_cols"]
    df_num = df[metadata["numeric_cols"]]

    # Scale numeric cols
    scaled = metadata["scaler"].transform(df_num)
    scaled_df = pd.DataFrame(scaled, columns=metadata["numeric_cols"], index=df.index)

    final = scaled_df.join(df[metadata["dummy_cols"]])
    final = final[final_cols]
    return final


# ---------- Streamlit UI ----------
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Predictor")
st.markdown("Use this simple app to predict median house value given feature inputs.")
# Set background image (uses `4house.png` from project root)
set_background("4house.png")

# Extract palette from same image and apply CSS theme
palette = extract_palette("4house.png", n_colors=5)
apply_palette_css(palette)

# Load and train
with st.spinner("Loading data and training models (runs once)..."):
    data = load_data("housing.csv")
    metadata = train_models(data)

# Model Info (displayed in main page)
# Determine best model by R^2 (higher is better). If equal, prefer Random Forest.
best_model_name = "Random Forest" if metadata["rf_score"] >= metadata["lr_score"] else "Linear Regression"
best_model = metadata["rf"] if best_model_name == "Random Forest" else metadata["lr"]

# Layout inputs in two columns
col1, col2 = st.columns(2)
with col1:
    longitude = st.number_input("Longitude", value=float(data["longitude"].median()))
    latitude = st.number_input("Latitude", value=float(data["latitude"].median()))
    housing_median_age = st.number_input("Housing median age", value=float(data["housing_median_age"].median()))
    total_rooms = st.number_input("Total rooms", value=float(data["total_rooms"].median()))

with col2:
    total_bedrooms = st.number_input("Total bedrooms", value=float(data["total_bedrooms"].median()))
    population = st.number_input("Population", value=float(data["population"].median()))
    households = st.number_input("Households", value=float(data["households"].median()))
    median_income = st.number_input("Median income (10k USD)", value=float(data["median_income"].median()))

ocean_options = list(data["ocean_proximity"].unique())
selected_ocean = st.selectbox("Ocean proximity", options=ocean_options)

# Assemble input row
input_dict = {
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [selected_ocean],
}
input_df = pd.DataFrame(input_dict)

if st.button("Predict"):
    processed = preprocess_input(input_df, metadata)
    model = best_model

    pred = model.predict(processed)[0]
    st.subheader("Predicted median house value")
    st.success(f"${pred:,.2f}")

    # If the chosen best model is Random Forest, show feature importances
    if best_model_name == "Random Forest":
        st.subheader("Feature importances (top 8)")
        importances = metadata["rf"].feature_importances_
        feats = metadata["numeric_cols"] + metadata["dummy_cols"]
        imp_df = pd.DataFrame({"feature": feats, "importance": importances}).sort_values("importance", ascending=False)
        st.table(imp_df.head(8).set_index("feature"))


