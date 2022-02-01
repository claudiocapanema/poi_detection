import numpy as np
import pandas as pd
import geopandas as gp
import timezonefinder
import pytz
import datetime as dt
import os
from contextlib import suppress

from configuration import BASE_DIR, CHECKINS, SPOTS_1, CATEGORY_STRUCTURE, TIMEZONES, BRASIL_STATES, COUNTRIES, \
    CHECKINS_5_CATEGORIES_OSM_LOCAL_DATETIME, CHECKINS_5_CATEGORIES_OSM

class DatetimesUtils:

    def __init__(self):
        self.tf = timezonefinder.TimezoneFinder()

    def date_from_str_to_datetime(self, date):
        pattern = '%Y-%m-%d %H:%M:%S'
        return dt.datetime.strptime(date, pattern)

    def convert_tz(self, datetime, from_tz, to_tz):
        datetime = datetime.replace(tzinfo=from_tz)
        datetime = datetime.astimezone(pytz.timezone(to_tz))
        return  datetime

    # @classmethod
    # def convert_tz(cls, datetime, from_tz, to_tz):
    #
    #     datetime = datetime.replace(tzinfo=from_tz)
    #     datetime = datetime.astimezone(pytz.timezone(to_tz)).to_pydatetime()
    #     year = datetime.year
    #     month = datetime.month
    #     day = datetime.day
    #     hour = datetime.hour
    #     minute = datetime.minute
    #     second = datetime.second
    #     # datetime = str(year)+"-"+str(month)+"-"+str(day)+" "+str(hour)+":"+str(minute)+":"+str(second)
    #     datetime = str(dt.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second))
    #     # datetime = datetime.srtftime("%Y-%m-%d %H:%M:%S")
    #     # .srtftime("%Y-%m-%d %H:%M:%S")
    #     return datetime

    def find_timezone(self, datetime, lat, lon):

        tf = timezonefinder.TimezoneFinder()
        timezone = tf.timezone_at(lng=lon, lat=lat)
        if timezone is None:
            print("Timezone inválido")
            return datetime
        if len(timezone) == 0:
            print("Timezone inválido")
            return datetime
        local_datetime = self.convert_tz(datetime, pytz.UTC, timezone)
        return local_datetime


if __name__ == "__main__":


    # df_states = gp.read_file(BRASIL_STATES)[['nome', 'geometry']]
    # df_states.columns = ['state_name', 'geometry']

    with suppress(OSError):
        os.remove(CHECKINS_5_CATEGORIES_OSM_LOCAL_DATETIME)

    datetime_utils = DatetimesUtils()
    first = True
    n = 0
    for df in pd.read_csv(CHECKINS_5_CATEGORIES_OSM, chunksize=5000000):

        #[['userid', 'placeid', 'datetime', 'latitude', 'longitude', 'category']]
        print("Describe datetime: ", df['datetime'].describe())
        df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%dT%H:%M:%SZ")
        df['index'] = np.array([i for i in range(len(df))])
        print("Tamanho inicial: ", len(df))
        gdf = gp.GeoDataFrame(
            df, geometry=gp.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

        timezones = gp.read_file(TIMEZONES)
        print("Timezones columns: ", timezones.columns)

        df_timezone = gp.sjoin(timezones, gdf, op='contains')[
            ['userid', 'index', 'placeid', 'datetime', 'tzid', 'latitude', 'longitude', 'category']]
        print("tamanho timezones: ", len(df_timezone))

        # countries
        df_countries = gp.read_file(COUNTRIES)
        df_countries = gp.sjoin(df_countries, gdf, op='contains')
        df_countries = df_countries[
            [ 'index', 'userid', 'placeid', 'datetime', 'latitude', 'longitude', 'CNTRY_NAME', 'category']]
        df_countries.columns = ['index', 'userid', 'placeid', 'datetime', 'latitude', 'longitude', 'country_name', 'category']
        df_countries = df_countries[['index', 'country_name']]
        print("tamanho countries: ", len(df_countries))

        # states
        df_states = gp.read_file(BRASIL_STATES)[['nome', 'geometry']]
        df_states.columns = ['state_name', 'geometry']
        df_states = gp.sjoin(df_states, gdf, op='contains')[['index', 'state_name']]

        datetime_list = df_timezone['datetime'].tolist()
        tz_list = df_timezone['tzid'].tolist()
        print("converter")
        local_datetime_list = []
        for i in range(len(datetime_list)):
            date = datetime_list[i]
            tz = tz_list[i]
            date = date.replace(tzinfo=pytz.utc)
            date = date.astimezone(pytz.timezone(tz))
            local_datetime_list.append(date)

        df_timezone['local_datetime'] = np.array(local_datetime_list)

        print("timezone")
        print(df_timezone.columns)
        print(df_timezone)
        print("countries")
        print(df_countries)
        print("events per country")
        events_per_country = \
            df_countries.groupby(by='country_name').apply(lambda e: pd.DataFrame({'Total events': [len(e)]})).reset_index()[
                ['country_name', 'Total events']].sort_values('Total events', ascending=False)
        print(events_per_country)

        df_timezone_country = df_timezone.join(df_countries.set_index('index'), on='index')
        df_timezone_country_state = df_timezone_country.join(df_states.set_index('index'), on='index')

        print("final")
        print("tamanho final (join): ", len(df_timezone_country_state))

        print(df_timezone_country_state.query("tzid == 'America/Sao_Paulo'")[['datetime', 'local_datetime']])

        df_timezone_country_state['country_name'] = df_timezone_country_state['country_name'].fillna('Others countries')

        state_name_list = df_timezone_country_state['state_name'].tolist()
        state_name_isnull_list = df_timezone_country_state['state_name'].isnull().tolist()
        country_name_list = df_timezone_country_state['country_name'].tolist()
        for i in range(len(state_name_isnull_list)):

            if state_name_isnull_list[i]:
                if country_name_list[i] != 'Others countries':
                    state_name_list[i] = "exterior_" + country_name_list[i]
                else:
                    state_name_list[i] = 'Others states'

        df_timezone_country_state['state_name'] = np.array(state_name_list)

        # print("events per country final")
        # events_per_country = \
        #     df_timezone_country_state.groupby(by='country_name').apply(
        #         lambda e: pd.DataFrame({'Total events': [len(e)]})).reset_index()[
        #         ['country_name', 'Total events']].sort_values('Total events', ascending=False)
        # print(events_per_country)
        #
        # print("events per state final")
        # events_per_country = \
        #     df_timezone_country_state.groupby(by='state_name').apply(
        #         lambda e: pd.DataFrame({'Total events': [len(e)]})).reset_index()[
        #         ['state_name', 'Total events']].sort_values('Total events', ascending=False)
        # print(events_per_country)
        #
        print(df_timezone_country_state)

        if first:
            df_timezone_country_state[
                ['userid', 'placeid', 'local_datetime', 'tzid', 'datetime', 'latitude', 'longitude', 'country_name',
                 'state_name', 'category']].to_csv(CHECKINS_5_CATEGORIES_OSM_LOCAL_DATETIME, index=False,)
            first = False
        else:
            df_timezone_country_state[
                ['userid', 'placeid', 'local_datetime', 'tzid', 'datetime', 'latitude', 'longitude', 'country_name',
                 'state_name', 'category']].to_csv(CHECKINS_5_CATEGORIES_OSM_LOCAL_DATETIME, index=False, mode='a', header=False)
