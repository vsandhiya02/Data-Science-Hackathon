# creating api endpoints using fastapi

# importing libraries
from fastapi import FastAPI
from pydantic import BaseModel  # pydantic - input and output will be in a structured format
import numpy as np
import pandas as pd
import joblib

import warnings
warnings.filterwarnings('ignore')

# create FastAPI object
app = FastAPI()

#to pass the input features
class Input(BaseModel):
  Gender                 :  object
  Age                    :   int
  Driving_License        :   int
  Region_Code            :   int
  Previously_Insured     :   int
  Vehicle_Age            :  object
  Vehicle_Damage         :  object
  Annual_Premium         : int
  Policy_Sales_Channel   : int
  Vintage                :   int

#to pass the output
class Output(BaseModel):
    Response: int

@app.post("/predict")
def predict(data: Input) -> Output: #data is a variable 
    # Creating the DataFrame from the input
    X_input = pd.DataFrame([[
  data.Gender,
  data.Age,
  data.Driving_License,
  data.Region_Code,
  data.Previously_Insured,
  data.Vehicle_Age,
  data.Vehicle_Damage,
  data.Annual_Premium,
  data.Policy_Sales_Channel,
  data.Vintage  
    ]])

    # Setting the columns names
    X_input.columns = [
        
    'Gender','Age','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Annual_Premium','Policy_Sales_Channel','Vintage']

    # Load the model
    model = joblib.load('insurance_model.pkl')

    # Predict
    prediction = model.predict(X_input)

    # Return the result
    return Output(Response=prediction)  # returning the predicted value