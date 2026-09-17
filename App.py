import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('linear_reg_model.sav')

st.title('Sales Prediction App')
st.write('Enter the advertising budgets to predict sales.')

# Input fields for TV, Radio, and Newspaper advertising budgets
tv = st.slider('TV Advertising Budget ($)', 0.0, 300.0, 150.0)
radio = st.slider('Radio Advertising Budget ($)', 0.0, 50.0, 25.0)
newspaper = st.slider('Newspaper Advertising Budget ($)', 0.0, 120.0, 30.0)

if st.button('Predict Sales'):
    # Create a numpy array from the inputs
    input_data = np.array([[tv, radio, newspaper]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    st.success(f'Predicted Sales: {prediction:.2f}')
