import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("Insurance Prediction")

# Install dependencies from requirements.txt
st.requirements("Hackathon/WebUI/requirements.txt")

#read the dataset to fill thevalues in input options of each element
df = pd.read_csv('train.csv')

#create the input elements
#categorical columns
Gender =st.selectbox("Gender", pd.unique(df['Gender']))
Vehicle_Age =st.selectbox("Vehicle_Age", pd.unique(df['Vehicle_Age']))
Vehicle_Damage =st.selectbox("Vehicle_Damage", pd.unique(df['Vehicle_Damage']))

#non-categorical columns
Age = st.number_input("Age")
Driving_License = st.number_input("Driving_License")
Region_Code = st.number_input("Region_Code")
Previously_Insured = st.number_input("Previously_Insured")
Annual_Premium = st.number_input("Annual_Premium")
Policy_Sales_Channel = st.number_input("Policy_Sales_Channel")
Vintage = st.number_input("Vintage")

#map the user inputs to respective column format

input = {
  'Gender' :Gender,
  'Age' :Age,
  'Driving_License' :Driving_License,
  'Region_Code' :Region_Code,
  'Previously_Insured' :Previously_Insured,
  'Vehicle_Age' :Vehicle_Age,
  'Vehicle_Damage' :Vehicle_Damage,
  'Annual_Premium' :Annual_Premium,
  'Policy_Sales_Channel' :Policy_Sales_Channel,
  'Vintage' :Vintage     
}

#load the model from the pickle file
model = joblib.load('insurance_model.pkl')

#action for the submit button
if st.button('Predict'):
    X_input = pd.DataFrame(input,index=[0])
    prediction = model.predict(X_input)
    st.write("The predicted value is:")
    st.write(prediction)
