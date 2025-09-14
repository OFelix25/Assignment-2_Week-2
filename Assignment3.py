import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load COVID-19 dataset
df = pd.read_csv(
    r"C:\Users\HP\Desktop\Felix Ochieng\CS\Felix Ochieng\Python\Data Analysis and Visualization\Assignment3\covid_19_clean_complete.csv"
)
# Handle missing values in confirmed, deaths, recovered
df[['Confirmed', 'Deaths', 'Recovered']] = df[['Confirmed', 'Deaths', 'Recovered']].fillna(0)

# Create new active cases feature
df['Active_Cases'] = df['Confirmed'] - df['Deaths'] - df['Recovered']

# Ensuring that Date is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Aggregating cases by country and date
country_grouped = df.groupby(['Country/Region', 'Date'])[['Confirmed', 'Deaths', 'Recovered', 'Active_Cases']].sum().reset_index()

# Daily new cases
country_grouped['Daily_New_Cases'] = country_grouped.groupby('Country/Region')['Confirmed'].diff().fillna(0).clip(lower=0)

# Normalize daily new cases using Min-Max
scaler = MinMaxScaler()
country_grouped['Daily_New_Cases_Norm'] = country_grouped.groupby('Country/Region')['Daily_New_Cases'].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1,1)).flatten()
)
# Plot normalized time-series for selected countries
plt.figure(figsize=(12, 6))
selected_countries = ['US', 'India', 'Brazil', 'Russia', 'United Kingdom']
for country in selected_countries:
    data = country_grouped[country_grouped['Country/Region'] == country]
    if not data.empty:
        plt.plot(data['Date'], data['Daily_New_Cases_Norm'], label=country)

plt.title("Normalized Daily New COVID-19 Cases")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Discretize latest daily new cases into intervals
latest_date = country_grouped['Date'].max()
latest_data = country_grouped[country_grouped['Date'] == latest_date]

low_q, medium_q = latest_data['Daily_New_Cases'].quantile([0.33, 0.66])
latest_data['Risk_Level'] = latest_data['Daily_New_Cases'].apply(
    lambda x: 'low' if x <= low_q else ('medium' if x <= medium_q else 'high')
)

summary = latest_data[['Country/Region', 'Daily_New_Cases', 'Risk_Level']].reset_index(drop=True)

print("Assignment completed successfully!")
