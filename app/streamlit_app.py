import streamlit as st
import joblib
import numpy as np
import os

# Use below snippet for loading data & model on local machine
# project_dir = r"C:\Users\anike\DATA SCIENCE AND AI\ML Projects\smartbuy-predictive-analytics-for-laptop-prices"
# data_dir = os.path.join('data', 'processed')

# model_path = os.path.join(project_dir, 'models', 'model.joblib')
# df_path = os.path.join(project_dir, data_dir, 'df.joblib')
# preprocessor_path = os.path.join(project_dir, 'models', 'preprocessor.joblib')

# model = joblib.load(model_path, mmap_mode=None)
# df = joblib.load(df_path, mmap_mode=None)
# preprocessor = joblib.load(preprocessor_path, mmap_mode=None)

# Use below snippet for loading data & model on GitHub Repo & Streamlit Cloud
model = joblib.load('models/model.joblib', mmap_mode=None)
df = joblib.load('data/processed/df.joblib', mmap_mode=None)
preprocessor = joblib.load('models/preprocessor.joblib', mmap_mode=None)

st.title("💻SmartBuy: Laptop Price Wizard")
st.markdown("## Predict laptop prices based on your custom configuration")

# Company
Company = st.selectbox('Brand', sorted(df['company'].unique()))

# Type of Laptop
Type = st.selectbox('Type', sorted(df['type_name'].unique()))

# OS
OS = st.selectbox('Operating System', sorted(df['os'].unique()))

# GPU Brand
GPU_company = st.selectbox('GPU Brand', sorted(df['gpu_company'].unique()))

# CPU Brand
CPU_brand = st.selectbox('CPU Brand', sorted(df['cpu_model'].unique()))

# Screen
Screen = st.selectbox('Display Quality', df['screen_type'].unique())

# Touchscreen
Touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])

# IPS Panel
IPSpanel = st.selectbox('IPS Panel', ['Yes', 'No'])

# Screen Size
Inches = st.number_input('Screen Size of the Laptop (in Inches)', min_value=0.01)

# Resolution
Resolution = st.selectbox('Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# Primary Storage
SSD = st.selectbox('SSD (in GB)', sorted(df['ssd'].unique()))

# Secondary Storage
HDD = st.selectbox('HDD (in GB)', sorted(df['hdd'].unique()))

# Secondary Storage
Flash_Storage = st.selectbox('Flash Storage (in GB)', sorted(df['flash_storage'].unique()))

# Secondary Storage
Hybrid = st.selectbox('Hybrid (in GB)', sorted(df['hybrid'].unique()))

# RAM Size
RAM = st.selectbox('RAM (in GB)', [2,4,6,8,12,16,24,32,64])

# Weight
Weight = st.number_input('Weight of the Laptop (in Kg)', min_value=0.01)

# CPU Frequency
CPU_freq = st.selectbox('CPU Frequency (in GHz)', sorted(df['cpu_freq'].unique()))

if st.button('Predict Price'):
    if Touchscreen == 'Yes':
        Touchscreen = 1
    else:
        Touchscreen = 0 

    if IPSpanel == 'Yes':
        IPSpanel = 1
    else:
        IPSpanel = 0

    X_res = int(Resolution.split('x')[0])
    Y_res = int(Resolution.split('x')[1])
    PPI = (((X_res**2) + (Y_res**2))**0.5) /(Inches)

    # Query
    query = np.array([[Company, Type, OS, GPU_company, CPU_brand, Screen, Touchscreen, IPSpanel, PPI, SSD, HDD, Flash_Storage, Hybrid, RAM, Weight, PPI]])
    
    query = query.reshape(1,16)
    query = preprocessor.transform(query)
    prediction = str(round(np.exp(model.predict(query))[0]))
    st.header("SmartBuy says: ₹" + prediction + " for this tech combo!")