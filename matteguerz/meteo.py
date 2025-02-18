import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from meteostat import Point, Daily
from prophet import Prophet
import plotly.express as px

def get_weather_data():
    # Tokyo coordinates
    city = Point(35.652832, 139.839478, 20)
    start = datetime(2022, 1, 1)
    end = datetime(2025, 2, 3)
    
    # Fetch data
    data = Daily(city, start, end).fetch()
    data = data.reset_index()
    return data

def prepare_prophet_data(data, target_col):
    # Prepare data for Prophet
    prophet_data = data[['time', target_col]].copy()
    prophet_data.columns = ['ds', 'y']
    prophet_data = prophet_data.dropna()
    return prophet_data

def make_forecast(data, target_col, days=7):
    # Prepare data
    prophet_data = prepare_prophet_data(data, target_col)
    
    # Create and fit model
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_data)
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    
    return forecast

def main():
    st.title("Tokyo Weather Forecast")
    
    # Get data
    data = get_weather_data()
    
    # Sidebar
    st.sidebar.header("Forecast Settings")
    forecast_days = st.sidebar.slider("Forecast Days", 1, 14, 7)
    target_column = st.sidebar.selectbox(
        "Select Weather Parameter",
        ['temp', 'prcp', 'wspd', 'humidity']
    )
    
    # Make forecast
    forecast = make_forecast(data, target_column, forecast_days)
    
    # Display historical data
    st.subheader("Historical Data")
    fig_historical = px.line(data, x='time', y=target_column,
                           title=f'Historical {target_column} in Tokyo')
    st.plotly_chart(fig_historical)
    
    # Display forecast
    st.subheader("Forecast")
    fig_forecast = px.line(forecast, x='ds', y=['yhat', 'yhat_lower', 'yhat_upper'],
                          title=f'{target_column} Forecast for Next {forecast_days} Days')
    st.plotly_chart(fig_forecast)
    
    # Display forecast metrics
    st.subheader("Forecast Values")
    forecast_tail = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
    forecast_tail.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
    st.dataframe(forecast_tail)

if __name__ == "__main__":
    main()