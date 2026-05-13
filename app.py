"""
Streamlit Web Application for Warsaw Real Estate Predictor.
Provides an interactive UI for users to input apartment parameters, 
reconstructs the one-hot encoded feature vector, and serves real-time 
inference from the serialized Random Forest model.
"""

import streamlit as st
import pandas as pd
import joblib

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Warsaw Rent Predictor", page_icon="🏙️", layout="centered")
st.title("🏙️ Warsaw Fair Market Rent Estimator")
st.markdown("""
This tool utilizes a **Random Forest Regressor** to estimate the fair monthly rental price 
of apartments in Warsaw based on real-time market data.
""")

# --- ASSET LOADING & CACHING ---
@st.cache_resource
def load_assets():
    """
    Loads serialized model and feature layout. 
    Cached to prevent memory overhead on subsequent UI interactions.
    """
    model = joblib.load('warsaw_rent_model.pkl')
    features = joblib.load('model_features.pkl')
    
    # Dynamically extract and alphabetize district names from the feature array
    districts = [f.replace('Distr_', '') for f in features if f.startswith('Distr_')]
    districts.sort() 
    
    return model, features, districts

try:
    rf_model, expected_features, available_districts = load_assets()

    # --- USER INTERFACE (SIDEBAR) ---
    st.sidebar.header("Apartment Parameters")
    
    # Input widgets
    size = st.sidebar.slider("Size (m²)", min_value=15, max_value=150, value=45, step=1)
    metro_dist = st.sidebar.slider("Distance to Metro (km)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    district = st.sidebar.selectbox("Select District", available_districts)

    # --- PORTFOLIO CREDIT (NEW) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("👨‍💻 About This Project")
    st.sidebar.info("""
    This application is an end-to-end Machine Learning portfolio project. 
    It features a custom web scraper, geospatial feature engineering, and a Random Forest prediction pipeline.
    
    * **Created by:** [Radosław Kowal]
    * **Code and Data:** [GitHub Repository](https://github.com/kowalradek/Warsaw-Rent-Calculator-Portfolio-Project)
    * **Connect:** [LinkedIn Profile](https://www.linkedin.com/in/rados%C5%82aw-kowal-098663300?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BK6ICEvDeTOuc4F2xNjoi0g%3D%3D)
    """)

    # --- INFERENCE ENGINE ---
    if st.button("Calculate Estimated Rent", type="primary"):
        
        # 1. Initialize a zeroed-out dictionary matching the model's exact feature layout
        input_data = {feat: 0 for feat in expected_features}
        
        # 2. Populate continuous numerical features
        input_data['Size_m2'] = size
        input_data['Dist_to_Metro_km'] = metro_dist
        
        # 3. Apply One-Hot Encoding for the categorical district selection
        selected_district_col = f"Distr_{district}"
        if selected_district_col in input_data:
            input_data[selected_district_col] = 1
            
        # 4. Cast to DataFrame for model ingestion
        input_df = pd.DataFrame([input_data])
        
        # 5. Execute prediction
        prediction = rf_model.predict(input_df)[0]
        price_per_sqm = prediction / size
        
        # --- DISPLAY RESULTS ---
        st.success(f"### Estimated Base Rent: **{int(prediction):,} PLN** / month")
        st.info(f"**Appraisal Breakdown:** {size} m² in {district} ({metro_dist} km to Metro) | Approx. **{int(price_per_sqm)} PLN/m²**")
        st.caption("⚠️ *Disclaimer: Estimates reflect base market rent and may not include administrative building fees (czynsz administracyjny) or utilities.*")

except FileNotFoundError:
    st.error("System Error: Model assets not found. Ensure 'warsaw_rent_model.pkl' and 'model_features.pkl' are located in the deployment directory.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
