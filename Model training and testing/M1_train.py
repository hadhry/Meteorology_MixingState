# trainM1.py

# Importing packages

print('Importing packages')

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# local time conversion
import pytz
from datetime import datetime
from timezonefinder import TimezoneFinder

from flaml import AutoML
from flaml.automl.data import get_output_from_log
import lightgbm as lgb
import pickle


# File import and formatting

print('Importing dataset')

df = xr.open_dataset('ERA5_3year.nc')

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

# Timezone conversion

print('Converting to local time')

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

# Formatting

df['local_time'] = pd.to_datetime(df['local_time'])

df = df[df['local_time'].dt.time == pd.to_datetime('6:00:00').time()]
df.loc[:, 'day_of_year'] = df['local_time'].dt.dayofyear

df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['t2m'] = pd.to_numeric(df['t2m'], errors='coerce')

X_train = df[['longitude', 'latitude', 'day_of_year']]
y_train = df['t2m']

X_train = X_train.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Model fit
print('starting model fit')

m = AutoML()
m.fit(X_train, y_train, task='regression', eval_method="cv", metric='rmse', time_budget=300, estimator_list=['lgbm'],early_stop=True, log_file_name='output/model1.log')

best_estimator = (f"{m.best_estimator}\n{m.best_config}")

with open('output/bestModel1.txt', "w") as file:
    file.write(best_estimator)

time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history = \
    get_output_from_log(filename='output/model1.log', time_budget= 300)


# To be ued for quantile regression models
m1_config = m.best_config

# Quantile models

print('starting quantile model fitting')
lower = lgb.LGBMRegressor(**m1_config, objective = 'quantile', alpha = 1 - 0.95)
lower.fit(X_train, y_train)

upper = lgb.LGBMRegressor(**m1_config, objective = 'quantile', alpha = 0.95)
upper.fit(X_train, y_train)

# Saving models
print('saving models')
pickle.dump(upper, open("output/model1Upper.pkl", "wb"))
pickle.dump(lower, open("output/model1Lower.pkl", "wb"))

# Finish
print("Training model 1 complete")


