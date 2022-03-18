

#Importing Dataset & Libraries

from importlib.resources import path
import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np

#App Customisation


st.header("Fish Weight Prediction App")
st.text_input("Enter your name:",key="name")

path = "/home/imman/Imman Codings/Deployment/Fish Weight Project/Resources/Fish.csv"
df = pd.read_csv(path)

#Load saved label encoder classes
encoder = LabelEncoder()

if st.checkbox('Show examples'):
    df




st.subheader("Please selecet revelant features of your fish!")
left_column,right_column = st.columns(2)
with left_column:
    inp_species = st.radio(
        'Name of the fish:',np.unique(df['Species'])
    )


input_length1 = st.slider('Vertical Length(cm)',0.0,
max(df['Length1']),1.0)

input_length2 = st.slider('Height(cm)',0.0,
max(df['Height']),1.0)

if st.button('Make Prediction'):
    print("THank you ! hope you like it ")
    st.success("Thank You")
    
st.sidebar.selectbox("What is your current background",("Student","Programmer",'Completely No'))
