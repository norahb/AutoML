import streamlit as st
from operator import index
import pandas as pd
import os
import matplotlib as plt
import plotly.express as px

# Profiling libraries
import pandas_profiling
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

#  ML libraries
from pycaret.regression import setup, compare_models, pull, save_model, load_model
from pycaret.classification import setup, compare_models, pull, save_model, load_model

with st.sidebar:
    st.image('https://cdn-icons-png.flaticon.com/512/3222/3222625.png')
    st.title('Automatic Machine Learning')
    choice = st.radio('Steps', ['Upload Data', 'Analyze Data', 'Train Model', 'Download Model'])
    st.info('This applicaiton allows you to automatically explore your dataset and train/compare various ML models  using Streamlit, Pandas Profiling and PyCaret')

if os.path.exists('sorucedata.csv'):
    df = pd.read_csv('sorucedata.csv', index_col=None)

if choice == 'Upload Data':
    st.title('Upload Your Data for Modeling')
    file = st.file_uploader('Upload your dataset here (csv)')
    if(file):
        df = pd.read_csv(file, index_col=None)
        df.to_csv('sorucedata.csv', index=None)
        st.dataframe(df)

if choice == 'Analyze Data':
    st.title('Automated Exploratory Data Analysis')
    profile_report = ProfileReport(df, title="Profiling Report")
    st_profile_report(profile_report)

if choice == 'Train Model':
    st.title('Machine Learning Analysis') 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    problem = st.selectbox('Select Your Problem Type', ('Classification', 'Regression'))
    if st.button('Train Models'):
        if problem == 'Classification':
            setup(data=df, target = chosen_target)
            best_model = compare_models()
            
        elif problem == 'Regression':
            setup(data=df, target = chosen_target, session_id=123)
            best_model = compare_models()
        
        # Display the results
        compare_df = pull()
        st.info('Machine learning models')
        st.dataframe(compare_df)
        st.write(best_model)
        save_model(best_model, 'best_model')

if choice == 'Download Model':
    st.title('Download The Trained Model')
    with open('best_model.pkl','rb') as f:
        st.download_button('Download the Model', f, 'trained_model.pkl')
