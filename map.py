import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="test")

uploaded_file = st.file_uploader('CSVファイルをアップロードしてください', type='csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['latitude'] = None
    df['longitude'] = None

    for index, row in df.iterrows():
        location = geolocator.geocode(row['住所'])
        df.at[index, 'latitude'] = location.latitude
        df.at[index, 'longitude'] = location.longitude

    df.to_csv('住所_緯度経度.csv', index=False)
    
df1 = pd.read_csv('住所_緯度経度.csv')

st.map(df1)