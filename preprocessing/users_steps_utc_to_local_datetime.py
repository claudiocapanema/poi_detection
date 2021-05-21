import numpy as np
import pandas as pd
from configuration import USERS_10_MIL_MAX_500_POINTS, COUNTRIES, USERS_10_MIL_MAX_500_POINTS_LOCAL_DATETIME, TIMEZONES, BRASIL_STATES
import timezonefinder
import pytz
import datetime as dt
import geopandas as gp

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

    df_states = gp.read_file(BRASIL_STATES)[['nome', 'geometry']]
    df_states.columns = ['state_name', 'geometry']

    datetime_utils = DatetimesUtils()
    df = pd.read_csv(USERS_10_MIL_MAX_500_POINTS)
    df['reference_date'] = pd.to_datetime(df['reference_date'], format="%Y-%m-%d %H:%M:%S")
    df['index'] = np.array([i for i in range(len(df))])
    print("Tamanho inicial: ", len(df))
    gdf = gp.GeoDataFrame(
        df, geometry=gp.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

    timezones = gp.read_file(TIMEZONES)
    print("Timezones columns: ", timezones.columns)

    df_timezone = gp.sjoin(timezones, gdf, op='contains')[['id', 'index', 'installation_id', 'reference_date', 'tzid', 'latitude', 'longitude']]
    print("tamanho timezones: ", len(df_timezone))

    # countries
    df_countries = gp.read_file(COUNTRIES)
    df_countries = gp.sjoin(df_countries, gdf, op='contains')
    df_countries = df_countries[['id', 'index', 'installation_id', 'reference_date', 'latitude', 'longitude', 'CNTRY_NAME']]
    df_countries.columns = ['id', 'index', 'installation_id', 'reference_date', 'latitude', 'longitude', 'country_name']
    df_countries = df_countries[['index', 'country_name']]
    print("tamanho countries: ", len(df_countries))

    # states
    df_states = gp.sjoin(df_states, gdf, op='contains')[['index', 'state_name']]

    datetime_list = df_timezone['reference_date'].tolist()
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
    df_countries.groupby(by='country_name').apply(lambda e: pd.DataFrame({'Total events': [len(e)]})).reset_index()[['country_name', 'Total events']].sort_values('Total events', ascending=False)
    print(events_per_country)

    df_timezone_country = df_timezone.join(df_countries.set_index('index'), on='index')
    df_timezone_country_state = df_timezone_country.join(df_states.set_index('index'), on='index')

    print("final")
    print("tamanho final (join): ", len(df_timezone_country_state))

    print(df_timezone_country_state.query("tzid == 'America/Sao_Paulo'")[['reference_date', 'local_datetime']])

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

    print("events per country final")
    events_per_country = \
        df_timezone_country_state.groupby(by='country_name').apply(lambda e: pd.DataFrame({'Total events': [len(e)]})).reset_index()[
            ['country_name', 'Total events']].sort_values('Total events', ascending=False)
    print(events_per_country)

    print("events per state final")
    events_per_country = \
        df_timezone_country_state.groupby(by='state_name').apply(
            lambda e: pd.DataFrame({'Total events': [len(e)]})).reset_index()[
            ['state_name', 'Total events']].sort_values('Total events', ascending=False)
    print(events_per_country)
    #
    print(df_timezone_country_state)
    df_timezone_country_state[['id', 'installation_id', 'local_datetime', 'tzid', 'reference_date', 'latitude', 'longitude', 'country_name', 'state_name']].to_csv(USERS_10_MIL_MAX_500_POINTS_LOCAL_DATETIME, index=False)


    # print(gf['reference_date'])
    # reference_date = gf['reference_date'].tolist()
    # latitude = gf['latitude'].tolist()
    # longitude = gf['longitude'].tolist()
    # countries = gf['country_name'].tolist()
    # print("Fora do brasil: ", len(gf.query("country_name != 'Brazil'")))
    # print("rodar")
    #
    # local_datetimes = []
    # p = int(len(gf)/10)
    # n = 10
    # for i in range(len(gf)):
    #     date = reference_date[i]
    #     if countries[i] == "Brazil":
    #         local_datetime = datetime_utils.convert_tz(date, pytz.utc, 'America/Sao_Paulo')
    #         local_datetimes.append(local_datetime)
    #     else:
    #         lat = latitude[i]
    #         lon = longitude[i]
    #         local_datetime = datetime_utils.find_timezone(date, lat, lon)
    #         local_datetimes.append(local_datetime)
    #     if i == p:
    #         p = p + p
    #         print("Roudou " + str(n) + "%")
    #         n = n + 10

    # df_countries['local_datetime'] = np.array(local_datetimes)
    #
    # df_countries[['id', 'installation_id', 'local_datetime', 'reference_date', 'latitude', 'longitude', 'country_name']].to_csv(USERS_10_MIL_MAX_500_POINTS_LOCAL_DATETIME, index=False)