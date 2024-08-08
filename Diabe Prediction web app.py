#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 06:36:59 2024

@author: lydiacharif
"""
import numpy as np
import pickle 
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('/Users/lydiacharif/PycharmProjects/diab_pred/trained_model.sav', 'rb'))

def diabe_prediction(input_data):
    input_data = [int(x) for x in input_data]
    input_data_npr = np.asarray(input_data)
    input_data_reshaped = input_data_npr.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person does not have diabetes'
    else:
        return 'The person has diabetes sadly hh'

def main():
    st.title('Diabetes Prediction')
    
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin")
    BMI = st.text_input("BMI")
    Age = st.text_input("Age")

    if st.button('Diabetes Test Results'):
        diagnosis = diabe_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age])
        st.success(diagnosis)

if __name__ == '__main__':
    main()

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
     