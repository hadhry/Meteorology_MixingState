# testM1.py

# Importing packages

print('importing packages')

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pytz
from datetime import datetime
from timezonefinder import TimezoneFinder

import pickle

from sklearn.metrics import mean_pinball_loss

# Importing dataset

print('importing test')

df = xr.open_dataset('ERA5_2022.nc')

#Convert longitude convention
df = df.assign_coords(longitude=(((df.longitude + 180) % 360) - 180))
df = df.sortby(df.longitude)  # Sort the data by longitude values

# Specify the new resolution
new_resolution = 2

# Nearest-neighbor subsampling
df = df.isel(longitude=slice(None, None, new_resolution), latitude=slice(None, None, new_resolution))

# Convert to pandas
df = df.to_dataframe()

df.reset_index(inplace=True)
df['time'] = pd.to_datetime(df['time'])

print('local time conversion')

def get_unique_locations(df):
    unique_locations = df[['latitude', 'longitude']].drop_duplicates()
    return unique_locations

def get_timezone(latitude, longitude, timezone_dict):
    if (latitude, longitude) not in timezone_dict:
        tf = TimezoneFinder()
        timezone = tf.timezone_at(lat=latitude, lng=longitude)
        timezone_dict[(latitude, longitude)] = timezone
    return timezone_dict[(latitude, longitude)]

def convert_utc_to_local(utc_time, timezone):
    utc_datetime = datetime.strptime(str(utc_time), '%Y-%m-%d %H:%M:%S')
    utc_timezone = pytz.timezone('UTC')
    local_timezone = pytz.timezone(timezone)
    local_datetime = utc_timezone.localize(utc_datetime).astimezone(local_timezone)
    return local_datetime.replace(tzinfo=None)


unique_locations = get_unique_locations(df)
timezone_dict = {}

# Calculate timezones for unique locations
for _, row in unique_locations.iterrows():
    timezone = get_timezone(row['latitude'], row['longitude'], timezone_dict)
    timezone_dict[(row['latitude'], row['longitude'])] = timezone

# Perform timezone conversion for all rows
df['local_time'] = df.apply(lambda row: convert_utc_to_local(row['time'], timezone_dict[(row['latitude'], row['longitude'])]), axis=1)

df['local_time'] = pd.to_datetime(df['local_time'])

df = df[df['local_time'].dt.time == pd.to_datetime('6:00:00').time()]
df.loc[:, 'day_of_year'] = df['local_time'].dt.dayofyear

df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['t2m'] = pd.to_numeric(df['t2m'], errors='coerce')

X_test = df[['longitude', 'latitude', 'day_of_year']]
y_test = df['t2m']

X_test = X_test.apply(pd.to_numeric, errors='coerce')
y_test = y_test.apply(pd.to_numeric, errors='coerce')

# Loading models

print('loading models')

upper = pickle.load(open('output/model1Upper.pkl', 'rb'))
lower = pickle.load(open('output/model1Lower.pkl', 'rb'))

#Predictions

print('making predictions')

lower_pred = lower.predict(X_test)
upper_pred = upper.predict(X_test)

upperscore = mean_pinball_loss(y_test, upper_pred, alpha=0.95)
lowerscore = mean_pinball_loss(y_test, lower_pred, alpha=0.05)
test_score = (f"Mean pinball loss q=0.95: {upperscore}\nMean pinball loss q=0.05: {lowerscore}")

with open('output/model1test_score.txt', "w") as file:
    file.write(test_score)

results = df[["longitude", "latitude", "day_of_year", "t2m"]].rename(columns={"t2m": "actual_t2m"})
results["upper_t2m"] = upper_pred
results["lower_t2m"] = lower_pred


# Loading the train

print('loading train for visualization')

data = xr.open_dataset('ERA5_3year.nc')

#Convert longitude convention
data = data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180))
data = data.sortby(data.longitude)  # Sort the data by longitude values


for i in range (3):
    # VISUALIZATION
    # Select a random row index
    random_row_index = results.sample().index[0]

    # Get the latitude and longitude from the random row
    vis_lat = results.loc[random_row_index, 'latitude']
    vis_long = results.loc[random_row_index, 'longitude']

    df = data.where((data['latitude'] == vis_lat) & (data['longitude'] == vis_long), drop=True)
    df = df.to_dataframe()

    df.reset_index(inplace=True)
    df['time'] = pd.to_datetime(df['time'])

    # Time conversion

    print('time conversion')

    def get_unique_locations(df):
        unique_locations = df[['latitude', 'longitude']].drop_duplicates()
        return unique_locations

    def get_timezone(latitude, longitude, timezone_dict):
        if (latitude, longitude) not in timezone_dict:
            tf = TimezoneFinder()
            timezone = tf.timezone_at(lat=latitude, lng=longitude)
            timezone_dict[(latitude, longitude)] = timezone
        return timezone_dict[(latitude, longitude)]

    def convert_utc_to_local(utc_time, timezone):
        utc_datetime = datetime.strptime(str(utc_time), '%Y-%m-%d %H:%M:%S')
        utc_timezone = pytz.timezone('UTC')
        local_timezone = pytz.timezone(timezone)
        local_datetime = utc_timezone.localize(utc_datetime).astimezone(local_timezone)
        return local_datetime.replace(tzinfo=None)


    unique_locations = get_unique_locations(df)
    timezone_dict = {}

    # Calculate timezones for unique locations
    for _, row in unique_locations.iterrows():
        timezone = get_timezone(row['latitude'], row['longitude'], timezone_dict)
        timezone_dict[(row['latitude'], row['longitude'])] = timezone

    # Perform timezone conversion for all rows
    df['local_time'] = df.apply(lambda row: convert_utc_to_local(row['time'], timezone_dict[(row['latitude'], row['longitude'])]), axis=1)

    df['local_time'] = pd.to_datetime(df['local_time'])

    df = df[df['local_time'].dt.time == pd.to_datetime('6:00:00').time()]
    df.loc[:, 'day_of_year'] = df['local_time'].dt.dayofyear

    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['t2m'] = pd.to_numeric(df['t2m'], errors='coerce')

    # Plotting


    # # Filter the DataFrame based on one location for visualization purposes
    filtered_results = results[(results['longitude'] == vis_long) & (results['latitude'] == vis_lat)]

    # # Get unique years present in the filtered DataFrame
    years = df['local_time'].dt.year.unique()

    # Plot all the years' data on the same plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    for year in years:
        year_data = df[df['local_time'].dt.year == year]
        plt.plot(year_data['day_of_year'], year_data['t2m'], label=f'Year {year}', alpha=0.25)
        
    plt.plot(filtered_results['day_of_year'], filtered_results['actual_t2m'], label='Actual t2m', color='orange')
    plt.plot(filtered_results['day_of_year'], filtered_results['upper_t2m'], label='Upper t2m', color = 'green')
    plt.plot(filtered_results['day_of_year'], filtered_results['lower_t2m'], label='Lower t2m', color = 'red')

    plt.xlabel('Day of Year')
    plt.ylabel('Temperature (t2m)')
    plt.title(f'6am temperature(t2m) at longitude {vis_long} and latitude {vis_lat}')
    plt.legend()
    plt.grid(True)  # Add a grid to the plot
    plt.savefig(f"output/m1visualization{i}.png")

    print(f'plot {i} complete')

print('Model 1 testing complete')



