import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.set_page_config(layout="wide")

st.title("Iris Plant Prediction app")

# reading all the pickle files
model_scaler = pickle.load(open('scaler_iris.pkl','rb')) # 1st pickle scale model
model_lr = pickle.load(open('model_iris.pkl','rb')) # 2nd pickle basic model


# user need to define the input
st.header("Enter the input values")

sepal_length=st.number_input(" Enter the float value for Sepal Lenght in cm **Min=4.3 Max=7.9**")
sepal_width=st.number_input(" Enter the float value for Sepal Width in cm **Min=2.0 Max=4.4**")
petal_length=st.number_input(" Enter the float value for Petal Lenght in cm **Min=1.0 Max=6.9**")
petal_width=st.number_input(" Enter the float value for Petal Width in cm **Min=0.1 Max=2.5**")

# create a dictionary for user_input
user_input={'sepal length (cm)':sepal_length,
    'sepal width (cm)':sepal_width,
    'petal length (cm)':petal_length,
    'petal width (cm)':petal_width}

# convert to Dataframe
user_input_df=pd.DataFrame(user_input,index=[0])


# scale the user_data
user_input_df_scaled=model_scaler.transform(user_input_df)

# Predict the Iris species
prediction = model_lr.predict(user_input_df_scaled)

if st.button("Predict"):
   
    result = prediction[0]
    
    if result == 0:
        st.success("Predicted Iris species: Setosa")
    elif result == 1:
        st.success("Predicted Iris species: Versicolor")
    else:
        st.success("Predicted Iris species: Virginica")
