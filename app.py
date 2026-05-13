import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page configuration
st.set_page_config(
    page_title="Alcohol Intake Predictor",
    page_icon="🍺",
    layout="centered"
)

# Website title
st.title("🍺 Alcohol Intake Predictor")

# Attractive sentence
st.markdown("""
### Discover Your Estimated Alcohol Intake Instantly 🍷🥃

Analyze alcohol consumption levels based on beer, spirit, and wine servings with an interactive Streamlit website.
""")

# Display image
st.image("BeerImage.webp", use_container_width=True)

# Section heading
st.header("📋 Enter Consumption Details")

# Instruction
st.write("Fill in the serving details below and click Predict.")

# Input fields
beer_servings = st.number_input(
    "🍺 Beer Servings",
    min_value=0.0,
    value=50.0
)

spirit_servings = st.number_input(
    "🥃 Spirit Servings",
    min_value=0.0,
    value=20.0
)

wine_servings = st.number_input(
    "🍷 Wine Servings",
    min_value=0.0,
    value=10.0
)

# Predict button
if st.button("🔍 Predict Alcohol Intake"):

    # Create feature array
    features = np.array([[
        beer_servings,
        spirit_servings,
        wine_servings
    ]])

    # Scale input
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)

    # Balloons animation
    st.balloons()

    # Output
    st.success(
        f"Predicted Total Litres of Pure Alcohol: {round(prediction[0],2)}"
    )

   

# Footer
st.markdown("""
---

""")