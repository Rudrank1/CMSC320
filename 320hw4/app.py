import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib


def get_session_state():
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = True
        return False
    return True

BACKGROUND_COLORS = {
    'KNN': '#692e2e',
    'LinearRegression': '#692e4b',
    'RandomForest': '#2e4b69',
    'XGBoost': '#0d4a13'
}

MODELS = {
    'KNN': joblib.load('models/KNN.pkl'),
    'LinearRegression': joblib.load('models/LinearRegression.pkl'),
    'RandomForest': joblib.load('models/RandomForest.pkl'),
    'XGBoost': joblib.load('models/XGBoost.pkl')
}

df = pd.read_csv('datasets/filtered_dataset.csv')

st.sidebar.title("Model Selection and Performance")

selected_model = st.sidebar.radio("Select Model", list(MODELS.keys()))
SELECTED_MODEL = MODELS[selected_model]

st.sidebar.write("### Model Performance Metrics")

st.sidebar.write("**KNN**")
st.sidebar.code("""
MSE: 0.5678
MAE: 0.5913
RMSE: 0.7535
R^2: 0.3698
""")

st.sidebar.write("**Linear Regression**")
st.sidebar.code("""
MSE: 0.1005
MAE: 0.2417
RMSE: 0.3171
R^2: 0.8884
""")

st.sidebar.write("**Random Forest**")
st.sidebar.code("""
MSE: 0.0989
MAE: 0.2292
RMSE: 0.3145
R^2: 0.8902
""")

st.sidebar.write("**XGBoost**")
st.sidebar.code("""
MSE: 0.0967
MAE: 0.2299
RMSE: 0.3110
R^2: 0.8926
""")

st.title("Professor Rating Predictor")

st.write("### Enter Professor Information")
col1, col2 = st.columns(2)

with col1:
    professor_name = st.text_input("Professor Name *", key="name")
with col2:
    professor_slug = st.text_input("Professor Slug (Optional)", key="slug", value="" if st.session_state.get("name", "") else st.session_state.get("slug", ""))

if not professor_name:
    st.warning("Please enter the professor's name")
else:
    # Find professor data
    if professor_name:
        professor_data = df[df['name'].str.contains(professor_name, case=False, na=False)]
    else:
        professor_data = df[df['slug'].str.contains(professor_slug, case=False, na=False)]
    
    if len(professor_data) == 0:
        st.error("No professor found with the given information")
    else:
        # Display professor information
        st.write("### Professor Information")
        st.dataframe(professor_data[['name', 'slug', 'num_courses', 'num_reviews', 'sentiment', 'average_rating']])
        
        # Prepare features for prediction
        features = ['num_courses', 'num_reviews', 'sentiment']
        X = professor_data[features]
        
        # Add predict button
        if st.button("Predict Rating"):
            # Make prediction
            prediction = SELECTED_MODEL.predict(X)
            
            # Display prediction
            st.write("### Prediction")
            st.write(f"Predicted Average Rating: {prediction[0]:.2f}")
            st.write(f"Actual Average Rating: {professor_data['average_rating'].values[0]:.2f}")
            
            # Show residuals plot
            st.image(f'results/{selected_model}/residuals.png', use_column_width=True)

background_color = BACKGROUND_COLORS[selected_model]
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {background_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)
