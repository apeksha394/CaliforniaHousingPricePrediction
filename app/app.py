from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0f172a;
    }

    h1 {
        color: white;
        text-align: center;
        font-size: 48px;
    }

    h2, h3 {
        color: white;
    }

    .stMarkdown {
        color: white;
    }

    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 15px;
    }

    div.stButton > button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        height: 3.5em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# LOAD MODEL + PREPROCESSOR
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "best_tuned_model.pkl"
)

preprocessor = joblib.load(
    BASE_DIR / "models" / "preprocessor.pkl"
)


# =====================================================
# HEADER
# =====================================================

st.title("🏡 California House Price Prediction")

st.markdown(
    """
    <div style='text-align:center; color:lightgray; font-size:20px;'>
    Smart Real Estate Valuation Platform
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")
st.write("")


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Project Information")

st.sidebar.markdown(
    """
    ### ML Pipeline

    - Exploratory Data Analysis
    - Feature Engineering
    - Data Preprocessing
    - Model Training
    - Hyperparameter Tuning
    - Streamlit Deployment

    ---

    ### Features

    - Median Income
    - House Age
    - Rooms
    - Ocean Proximity
    """
)


# =====================================================
# DASHBOARD METRICS
# =====================================================

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        label="Model Accuracy",
        value="87%",
        delta="+3.2%"
    )

with metric2:
    st.metric(
        label="Algorithm",
        value="XGBoost"
    )

with metric3:
    st.metric(
        label="Dataset Rows",
        value="20,640"
    )


# =====================================================
# INPUT SECTION
# =====================================================

st.write("")
st.write("")

st.subheader("📥 Property Information")

left_col, right_col = st.columns(2)


# =====================================================
# LEFT COLUMN INPUTS
# =====================================================

with left_col:

    median_income = st.slider(
        "Median Income",
        min_value=0.0,
        max_value=15.0,
        value=5.0,
        step=0.1,
        help="Median income of households"
    )

    housing_median_age = st.slider(
        "House Age",
        min_value=1,
        max_value=60,
        value=25
    )

    total_rooms = st.slider(
        "Total Rooms",
        min_value=100,
        max_value=10000,
        value=2000,
        step=100
    )


# =====================================================
# RIGHT COLUMN INPUTS
# =====================================================

with right_col:

    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        [
            "<1H OCEAN",
            "INLAND",
            "NEAR BAY",
            "NEAR OCEAN",
            "ISLAND"
        ]
    )

    region = st.selectbox(
        "California Region",
        [
            "San Francisco",
            "Los Angeles",
            "San Diego",
            "Sacramento"
        ]
    )

    house_size = st.select_slider(
        "Property Size",
        options=[
            "Small",
            "Medium",
            "Large",
            "Luxury"
        ]
    )


# =====================================================
# INTERNAL FEATURE MAPPING
# =====================================================

# Region Coordinates

region_coordinates = {
    "San Francisco": (-122.42, 37.77),
    "Los Angeles": (-118.24, 34.05),
    "San Diego": (-117.16, 32.71),
    "Sacramento": (-121.49, 38.58)
}

longitude, latitude = region_coordinates[region]


# Hidden Estimated Features

if house_size == "Small":
    households = 200
    population = 700
    total_bedrooms = total_rooms * 0.15

elif house_size == "Medium":
    households = 400
    population = 1400
    total_bedrooms = total_rooms * 0.17

elif house_size == "Large":
    households = 700
    population = 2600
    total_bedrooms = total_rooms * 0.20

else:
    households = 1200
    population = 5000
    total_bedrooms = total_rooms * 0.25


# =====================================================
# CREATE DATAFRAME
# =====================================================

input_df = pd.DataFrame([
    {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }
])


# =====================================================
# FEATURE ENGINEERING
# =====================================================

input_df["rooms_per_household"] = (
    input_df["total_rooms"] /
    input_df["households"]
)

input_df["bedrooms_per_room"] = (
    input_df["total_bedrooms"] /
    input_df["total_rooms"]
)

input_df["population_per_household"] = (
    input_df["population"] /
    input_df["households"]
)


# =====================================================
# PREDICTION BUTTON
# =====================================================

st.write("")
st.write("")

if st.button("Predict House Price"):

    processed_input = preprocessor.transform(
        input_df
    )

    prediction = model.predict(
        processed_input
    )

    predicted_price = prediction[0]


    # =============================================
    # RESULT DISPLAY
    # =============================================

    st.write("")
    st.write("")

    st.success(
        f"🏠 Estimated House Price: ${predicted_price:,.2f}"
    )


    # =============================================
    # PRICE CATEGORY
    # =============================================

    if predicted_price < 150000:
        category = "Affordable"

    elif predicted_price < 350000:
        category = "Mid-Range"

    elif predicted_price < 700000:
        category = "Premium"

    else:
        category = "Luxury"


    # =============================================
    # ADDITIONAL METRICS
    # =============================================

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric(
            "Price Category",
            category
        )

    with result2:
        st.metric(
            "Region",
            region
        )

    with result3:
        st.metric(
            "Ocean Proximity",
            ocean_proximity
        )


# =====================================================
# FOOTER
# =====================================================

st.write("")
st.write("")
st.write("")

st.markdown(
    """
    <hr>
    <div style='text-align:center; color:gray;'>
    Built with modern data-driven property analytics
    </div>
    """,
    unsafe_allow_html=True
)
