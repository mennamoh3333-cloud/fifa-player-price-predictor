# player_price_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# ===============================
# Load Model
# ===============================
model = joblib.load("player_price_predictor.pkl")

st.set_page_config(page_title="Player Price Predictor", page_icon="⚽", layout="centered")

# ===============================
# Set Background Image
# ===============================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ضع الصورة في نفس فولدر المشروع
set_bg("stadium.jpg")

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("Player Attributes")
age = st.sidebar.slider("🎂 Age", 16, 45, 22)
overall = st.sidebar.slider("⭐ Overall Rating", 40, 99, 75)
potential = st.sidebar.slider("🚀 Potential Rating", 40, 99, 80)
pace = st.sidebar.slider("⚡ Pace", 1, 99, 70)
shooting = st.sidebar.slider("🎯 Shooting", 1, 99, 70)
passing = st.sidebar.slider("🎯 Passing", 1, 99, 70)
dribbling = st.sidebar.slider("🌀 Dribbling", 1, 99, 70)
defending = st.sidebar.slider("🛡 Defending", 1, 99, 50)
physic = st.sidebar.slider("💪 Physic", 1, 99, 70)

# ===============================
# Title
# ===============================
st.markdown(
    "<h1 style='text-align:center; color:white;'>⚽ FIFA Player Price Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:white;'>AI-powered football market value estimation</p>",
    unsafe_allow_html=True
)

# ===============================
# Prediction
# ===============================
if st.button("⚽ Predict Player Value"):
    input_data = pd.DataFrame({
        'age': [age],
        'overall': [overall],
        'potential': [potential],
        'pace': [pace],
        'shooting': [shooting],
        'passing': [passing],
        'dribbling': [dribbling],
        'defending': [defending],
        'physic': [physic]
    })

    log_price = model.predict(input_data)[0]
    price = np.expm1(log_price)

    # Format price
    if price >= 1_000_000:
        price_str = f"{price/1_000_000:.2f}M €"
    elif price >= 1_000:
        price_str = f"{price/1_000:.2f}K €"
    else:
        price_str = f"{price:.2f} €"

    # Display FIFA-style Card
    st.markdown(
        f"""
        <div style="
            margin-top: 25px;
            background: linear-gradient(135deg, #0b6623, #0f8a3a);
            padding: 24px;
            border-radius: 18px;
            color: white;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            max-width: 420px;
            margin-left: auto;
            margin-right: auto;
        ">
            <div style="font-size:22px; font-weight:700;">💰 Market Value</div>
            <div style="font-size:34px; font-weight:800; margin-top:10px;">
                {price_str}
            </div>
            <div style="font-size:14px; opacity:0.85; margin-top:8px;">
                FIFA-style estimation
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
