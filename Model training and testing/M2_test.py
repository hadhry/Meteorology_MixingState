# testM2.py

#Import packages
print("Importing packages")

import xarray as xr
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import pytz
from datetime import datetime
from timezonefinder import TimezoneFinder
import pickle


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# Reading dataset

print("Reading test dataset")

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
df[df.columns] = df[df.columns].astype(object)
df['time'] = pd.to_datetime(df['time'])

#Local time conversion
print("Converting to local time")

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

df.drop('time', axis=1, inplace=True)
df['local_time'] = pd.to_datetime(df['local_time'])

# FORMATTING
# Pivotting tables
print("Pivoting tables")

df['date'] = df['local_time'].dt.date
df['hour'] = df['local_time'].dt.hour

#Adjusting time so that 6am will be the first hour
df['hour'] = (df['hour'] - 6) % 24

df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')

# Shift 'local_time' back by one day for hours between 18 and 23
mask = (df['hour'] >= 18) & (df['hour'] <= 23)
df.loc[mask, 'date'] -= pd.Timedelta(days=1)

df['day_of_year'] = df['date'].dt.dayofyear

df = df.pivot_table(index=['longitude', 'latitude', 'date', 'day_of_year'], columns='hour', values='t2m')


df.reset_index(inplace=True)

# Fixing column names
def map_hour_to_am_pm(hour):
    if hour == 0:
        return "12am"
    elif 1 <= hour < 12:
        return f"{hour}am"
    elif hour == 12:
        return "12pm"
    else:
        return f"{hour-12}pm"

# Looping through columns
for col in df.columns:
    if isinstance(col, int):
        hour = (col + 6) % 24  # To handle the 24-hour cycle (+6 as 6am is the first temperature)
        new_col_name = map_hour_to_am_pm(hour)
        df.rename(columns={col: new_col_name}, inplace=True)

df = df.dropna()
df.drop('date', axis=1, inplace=True)
df.columns.name = None

print("Formatting complete")

X_test = df[['longitude', 'latitude', 'day_of_year', '6am']]
y_test = df.iloc[:, 4:]

X_test = X_test.apply(pd.to_numeric, errors='coerce')
y_test = y_test.apply(pd.to_numeric, errors='coerce')

#Loading model
print('loading model')

model = pickle.load(open('output/model2.pkl', 'rb'))

# Predicting
print("Making prediction")


y_pred = model.predict(X_test.values)

rmse = mean_squared_error(y_test, y_pred, squared=False)
rmseraw = mean_squared_error(y_test, y_pred, multioutput='raw_values')
r2 = r2_score(y_test, y_pred)
r2raw = r2_score(y_test, y_pred, multioutput='raw_values')
r2variance = r2_score(y_test, y_pred, multioutput='variance_weighted')
MAE = mean_absolute_error(y_test, y_pred)

test_score = (f"RMSE (RMSE): {rmse}\nRMSE RAW: {rmseraw}\nR2 Score: {r2}\nR2 variance: {r2variance}\nR2 raw: {r2raw}\nMAE: {MAE}")


with open('output/model2superraw.txt', "w") as file:
    file.write(test_score)

print("complete")






