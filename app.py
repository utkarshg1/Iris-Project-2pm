# Import all the packages
import streamlit as st
import pickle 
import pandas as pd
import numpy as np

# Build the user interface
st.set_page_config(page_title="Iris Project", layout='wide')

# Add title to the body
st.title("Iris Project - Utkarsh Gaikwad")

# Add inputs for user 
sep_len = st.number_input("Sepal Length : ", min_value=0.00 , step=0.01)
sep_wid = st.number_input("Sepal Width : ", min_value=0.00, step=0.01)
pet_len = st.number_input("Petal Length : ", min_value=0.00, step=0.01)
pet_wid = st.number_input("Petal Width : ",min_value=0.00, step=0.01)

# Add a button to predict
submit = st.button("Predict")

# Load the preprocessor with pickle
with open("notebook/pre.pkl", "rb") as file1:
    pre = pickle.load(file1)

# Load the model with pickle
with open('notebook/model.pkl', 'rb') as file2:
    model = pickle.load(file2)

# if submit button is pressed
if submit:
    # Convert the data into dataframe
    dct = {'sepal_length':[sep_len],
           'sepal_width':[sep_wid],
           'petal_length':[pet_len],
           'petal_width':[pet_wid]}
    # Convert above dictionary to dataframe
    xnew = pd.DataFrame(dct)
    # Transform xnew
    xnew_pre = pre.transform(xnew)
    # Predict the results with probability
    pred = model.predict(xnew_pre)
    prob = model.predict_proba(xnew_pre)
    max_prob = np.max(prob)
    # Print above results
    st.subheader("Predictions are : ")
    st.subheader(f"Predicted Species : {pred[0]}")
    st.subheader(f"Probability : {max_prob*100:.2f} %")
    st.progress(max_prob)