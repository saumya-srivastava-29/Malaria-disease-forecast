from cleandata import get_clean_data
import matplotlib.pyplot as plt

# ====== Load Cleaned Data ======
data_file = "API_SH.MLR.INCD.P3_DS2_en_csv_v2_27849.csv"
df = get_clean_data(data_file)

# ====== Filter for a Specific Country ======
selected_country = "India"
country_df = df[df["country"] == selected_country]

# ====== Plot Malaria Trend ======
plt.figure(figsize=(10, 5))
plt.plot(country_df["year"], country_df["incidence"], marker='o')
plt.title(f"Malaria Incidence in {selected_country} Over Time")
plt.xlabel("Year")
plt.ylabel("Incidence (per 1,000 at risk)")
plt.grid(True)
plt.tight_layout()
plt.show()

from prophet import Prophet

# Prepare data for Prophet
forecast_df = country_df.rename(columns={"year": "ds", "incidence": "y"})

# Fit the model
model = Prophet()
model.fit(forecast_df)

# Create future dates (next 5 years)
future = model.make_future_dataframe(periods=5, freq='YE')

# Predict
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title(f"Forecasted Malaria Incidence for {selected_country}")
plt.xlabel("Year")
plt.ylabel("Incidence (per 1,000)")
plt.tight_layout()
plt.show()

# Plot components: trend and seasonality
model.plot_components(forecast)
plt.tight_layout()
plt.show()

