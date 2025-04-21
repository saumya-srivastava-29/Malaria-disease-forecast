import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from cleandata import get_clean_data

# Load and clean data
df = get_clean_data("API_SH.MLR.INCD.P3_DS2_en_csv_v2_27849.csv")

# Sidebar: country selector
countries = sorted(df["country"].unique())
selected_country = st.sidebar.selectbox("🌍 Select a country:", countries)

# Filter country data
country_df = df[df["country"] == selected_country]

# Sidebar: date range filter
min_year = country_df["year"].dt.year.min()
max_year = country_df["year"].dt.year.max()
start_year, end_year = st.sidebar.slider(
    "📅 Select year range:",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Filter historical data by range
filtered_df = country_df[
    country_df["year"].dt.year.between(start_year, end_year)
]

# Title
st.title("🦟 Malaria Incidence Forecast")
st.subheader(f"📌 Country: {selected_country} ({start_year} to {end_year})")

# Plot filtered historical data
st.line_chart(filtered_df.set_index("year")["incidence"])

# Forecasting
st.subheader("🔮 Forecast for Next 5 Years")
forecast_df = country_df.rename(columns={"year": "ds", "incidence": "y"})

model = Prophet()
model.fit(forecast_df)
future = model.make_future_dataframe(periods=5, freq="YE")
forecast = model.predict(future)

# Plot forecast
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Plot components
st.subheader("📊 Trend & Seasonality")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# CSV Download
st.subheader("⬇️ Download Forecast")
download_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
download_df.columns = ["Date", "Predicted Incidence", "Lower Bound", "Upper Bound"]

csv = download_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Forecast as CSV",
    data=csv,
    file_name=f"{selected_country}_malaria_forecast.csv",
    mime='text/csv'
)
