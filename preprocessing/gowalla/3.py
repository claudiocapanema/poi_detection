import numpy as np
import pandas as pd
import geopandas as gp
import timezonefinder
import pytz
import datetime as dt
from contextlib import suppress
import os

from configuration import BASE_DIR, CHECKINS, SPOTS_1, CATEGORY_STRUCTURE, TIMEZONES, BRASIL_STATES, COUNTRIES, \
    CHECKINS_LOCAL_DATETIME, CHECKINS_7_CATEGORIES, CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED


if __name__ == "__main__":

    with suppress(OSError):
        os.remove(CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED)
    first = True
    for df in pd.read_csv(CHECKINS_LOCAL_DATETIME, chunksize=5000000):

        df = df[['userid', 'placeid', 'local_datetime', 'latitude', 'longitude', 'country_name', 'state_name', 'category']]
        df = df.dropna(subset=['category'])
        local_datetime_list = df['local_datetime'].tolist()
        for i in range(len(local_datetime_list)):
            local_datetime_list[i] = local_datetime_list[i][:19]
        df['local_datetime'] = np.array(local_datetime_list)
        print(df)

        print("categorias uunicas: ", df['category'].unique().tolist())



        if first:
            df.to_csv(CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED, index=False)
            first = False

        else:
            df.to_csv(CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED, index=False, mode='a', header=False)