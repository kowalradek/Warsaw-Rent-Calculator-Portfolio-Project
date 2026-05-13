import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Warsaw Rent Predictor", page_icon="🏙️")
st.title("🏙️ Warsaw Fair Market Rent Estimator")
st.markdown("""
This tool uses a **Random Forest Machine Learning model** to estimate the fair rental price 
of apartments in Warsaw based on real-time market data.
""")

# 2. Load the AI Model
@st.cache_resource
def load_assets():
    # Load the Random Forest model and the exact columns it was trained on
    model = joblib.load('warsaw_rent_model.pkl')
    features = joblib.load('model_features.pkl')
    
    # Extract just the district names for the dropdown menu
    districts = [f.replace('Distr_', '') for f in features if f.startswith('Distr_')]
    districts.sort() # Alphabetize for the user
    
    return model, features, districts

try:
    model, features, districts = load_assets()

    # 3. User Interface (Sidebar)
    st.sidebar.header("Apartment Features")
    size = st.sidebar.slider("Size (m²)", min_value=15, max_value=150, value=45, step=1)
    metro_dist = st.sidebar.slider("Distance to Metro (km)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    district = st.sidebar.selectbox("Select District", districts)

    # 4. Prediction Logic
    if st.button("Calculate Estimated Rent", type="primary"):
        # Create a dictionary where every column is set to 0
        input_data = {feat: 0 for feat in features}
        
        # Fill in the user's specific inputs
        input_data['Size_m2'] = size
        input_data['Dist_to_Metro_km'] = metro_dist
        
        # Switch the selected district's value to 1 (One-Hot Encoding)
        selected_district_col = f"Distr_{district}"
        if selected_district_col in input_data:
            input_data[selected_district_col] = 1
            
        # Convert to a DataFrame so the model can read it
        input_df = pd.DataFrame([input_data])
        
        # Make the prediction
        prediction = model.predict(input_df)[0]
        
        # 5. Display Results
        st.balloons()
        st.success(f"### Estimated Monthly Rent: **{int(prediction):,} PLN**")
        st.info(f"**Breakdown:** {size} m² in {district} ({metro_dist} km from Metro) | Approx. **{int(prediction/size)} PLN/m²**")
        st.caption("⚠️ *Estimates reflect base rent and may not include administrative fees (czynsz).*")

except Exception as e:
    st.error(f"System Error: {e}. Ensure the .pkl files are in the same folder as this script.")
