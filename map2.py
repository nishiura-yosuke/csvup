import streamlit as st
import pandas as pd
import pydeck as pdk
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="test")

uploaded_file = st.file_uploader('CSVファイルをアップロードしてください', type='csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['latitude'] = None
    df['longitude'] = None

    for index, row in df.iterrows():
        location = geolocator.geocode(row['住所'])
        if location is not None:
            df.at[index, 'latitude'] = location.latitude
            df.at[index, 'longitude'] = location.longitude

    df.to_csv('住所_緯度経度.csv', index=False)
    df1 = pd.read_csv('住所_緯度経度.csv')

    view_state = pdk.ViewState(
        latitude=df1['latitude'].mean(),
        longitude=df1['longitude'].mean(),
        zoom=11,
        pitch=0)


    
    layers=[
        pdk.Layer(
           'HexagonLayer',
           data=df1,
           get_position='[longitude, latitude]',
           radius=200,
           elevation_scale=4,
           elevation_range=[0, 500],
           pickable=True,
           extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=df1,
            get_position='[longitude, latitude]',
            get_color='[200, 30, 0, 160]',
            get_radius=200,
        ),
    ],

    app = pdk.Deck(layers=[layers], initial_view_state=view_state)
    st.pydeck_chart(app)
