import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests # Make sure this is imported

# --- Function to Download Files ---
def download_file(url, local_filename):
    """Downloads a file from a URL if it doesn't exist locally."""
    if not os.path.exists(local_filename):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(local_filename), exist_ok=True)
        st.info(f"Downloading {os.path.basename(local_filename)}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return local_filename

# --- Load the Saved Model and Objects ---
@st.cache_resource
def load_model():
    """Downloads and loads the saved model, scaler, and label encoder."""
    model_dir = "saved_models"

    # --- Your URLs are now in the code ---
    MODEL_URL = "https://github.com/Sharad0404/Exoplanets_NASA/releases/download/Exoplanet/exoplanet_stacking_model.joblib"
    SCALER_URL = "https://github.com/Sharad0404/Exoplanets_NASA/releases/download/Exoplanet/scaler.joblib"
    LE_URL = "https://github.com/Sharad0404/Exoplanets_NASA/releases/download/Exoplanet/label_encoder.joblib"
    
    # Download files to the 'saved_models' subdirectory
    model_path = download_file(MODEL_URL, os.path.join(model_dir, 'exoplanet_stacking_model.joblib'))
    scaler_path = download_file(SCALER_URL, os.path.join(model_dir, 'scaler.joblib'))
    le_path = download_file(LE_URL, os.path.join(model_dir, 'label_encoder.joblib'))

    # Load objects from the downloaded files
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    return model, scaler, le

# --- Your original app code starts here ---
model, scaler, le = load_model()

# --- Build the User Interface ---
st.title('Exoplanet Disposition Predictor 🪐')
st.write("This app uses a machine learning model to predict if a star's signal is a real exoplanet candidate.")
st.write("Enter the parameters of a signal in the sidebar to get a prediction.")

st.sidebar.header('Input Transit Features')

# Create number inputs in the sidebar for key features
koi_score = st.sidebar.number_input(
    'Disposition Score (koi_score)', 
    min_value=0.0, max_value=1.0, value=0.95, step=0.01,
    help="A score from 0 to 1 indicating the likelihood of being a planet."
)
koi_fpflag_nt = st.sidebar.selectbox(
    'Not Transit-Like Flag', [0, 1],
    help="A flag indicating the signal does not have a transit-like shape."
)
koi_fpflag_ss = st.sidebar.selectbox(
    'Stellar Eclipse Flag', [0, 1],
    help="A flag indicating the signal is likely from a background star eclipsing another."
)
koi_model_snr = st.sidebar.number_input(
    'Transit Signal-to-Noise (SNR)', 
    min_value=0.0, value=30.0, step=1.0,
    help="The strength of the transit signal compared to the noise."
)
koi_prad = st.sidebar.number_input(
    'Planetary Radius (Earth Radii)', 
    min_value=0.0, value=2.0, step=0.1,
    help="The estimated radius of the planet in multiples of Earth's radius."
)

# --- Preprocess User Input and Predict ---
if st.sidebar.button('Predict Disposition'):
    # Create a dictionary from the user's input
    # NOTE: We create a full feature set, filling defaults for features not in the UI
    feature_dict = {
        'koi_score': koi_score, 'koi_fpflag_nt': koi_fpflag_nt, 'koi_fpflag_ss': koi_fpflag_ss,
        'koi_model_snr': koi_model_snr, 'koi_prad': koi_prad,
        # Fill other features with typical or mean values for this demo
        'koi_period': 30.0, 'koi_duration': 5.0, 'koi_depth': 1000.0, 'koi_teq': 1200.0,
        'koi_insol': 100.0, 'koi_steff': 5500.0, 'koi_slogg': 4.5, 'koi_srad': 1.0,
        'koi_fpflag_co': 0, 'koi_fpflag_ec': 0, 'koi_impact': 0.5, 'koi_period_err1': 0.0005,
        'koi_duration_err1': 0.1, 'koi_impact_err1': 0.1, 'koi_steff_err1': 100.0
    }
    
    # Create the engineered features from the dictionary values
    feature_dict['transit_shape'] = feature_dict['koi_depth'] / (feature_dict['koi_duration'] + 1e-6)
    feature_dict['period_certainty'] = feature_dict['koi_period'] / (feature_dict['koi_period_err1'] + 1e-6)

    # Convert the dictionary to a pandas DataFrame
    features_df = pd.DataFrame([feature_dict])

    # Ensure the column order of the new data matches the training data
    training_cols = scaler.get_feature_names_out()
    features_df = features_df[training_cols]

    # Use the loaded scaler to transform the numeric features
    features_df[training_cols] = scaler.transform(features_df[training_cols])
    
    # Make the prediction and get probabilities
    prediction_encoded = model.predict(features_df)
    prediction_proba = model.predict_proba(features_df)
    prediction_readable = le.inverse_transform(prediction_encoded)[0]

    # Display the final prediction
    st.subheader(f'Prediction: **{prediction_readable}**')

    # Display the confidence probabilities in a bar chart
    st.write("Prediction Confidence:")
    proba_df = pd.DataFrame(prediction_proba, columns=le.classes_).T
    proba_df.columns = ['Confidence']
    st.bar_chart(proba_df)
