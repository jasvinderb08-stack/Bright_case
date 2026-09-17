import streamlit as st
import joblib
import numpy as np

import os
import joblib

# Resolve absolute path relative to App.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'linear_reg.sav')

model = joblib.load(model_path)
model = joblib.load(open('linear_reg.sav', 'rb'))

st.title('Sales Prediction App')

# Input features
TV = st.number_input('TV Advertising Budget', min_value=0.0)
Radio = st.number_input('Radio Advertising Budget', min_value=0.0)
Newspaper = st.number_input('Newspaper Advertising Budget', min_value=0.0)

# Make prediction
if st.button('Predict Sales'):
    input_data = np.array([[TV, Radio, Newspaper]])
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Sales: {prediction:.2f}')
