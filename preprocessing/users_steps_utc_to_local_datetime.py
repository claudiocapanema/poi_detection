import numpy as np
import pandas as pd
from configuration import USERS_10_MIL_MAX_500_POINTS, COUNTRIES, USERS_10_MIL_MAX_500_POINTS_LOCAL_DATETIME, TIMEZONES
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

    datetime_utils = DatetimesUtils()
    df = pd.read_csv(USERS_10_MIL_MAX_500_POINTS)
    df['reference_date'] = pd.to_datetime(df['reference_date'], infer_datetime_format=True)
    df['index'] = np.array([i for i in range(len(df))])
    print("Tamanho inicial: ", len(df))
    gdf = gp.GeoDataFrame(
        df, geometry=gp.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

    timezones = gp.read_file(TIMEZONES)
    print("Timezones columns: ", timezones.columns)

    df_timezone = gp.sjoin(timezones, gdf, op='contains')[['id', 'index', 'installation_id', 'reference_date', 'tzid', 'latitude', 'longitude']]

    df_countries = gp.read_file(COUNTRIES)
    df_countries = gp.sjoin(df_countries, gdf, op='contains')
    df_countries = df_countries[['id', 'index', 'installation_id', 'reference_date', 'latitude', 'longitude', 'CNTRY_NAME']]
    df_countries.columns = ['id', 'index', 'installation_id', 'reference_date', 'latitude', 'longitude', 'country_name']
    df_countries = df_countries[['index', 'country_name']]

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

    df_timezone_country = df_timezone.join(df_countries.set_index('index'), on='index')

    print("final")
    print(df_timezone_country)
    print("tamanho final: ", len(df_timezone_country))

    print(df_timezone_country.query("tzid == 'America/Sao_Paulo'")[['reference_date', 'local_datetime']])
    #
    df_timezone_country[['id', 'installation_id', 'local_datetime', 'tzid', 'reference_date', 'latitude', 'longitude', 'country_name']].to_csv(USERS_10_MIL_MAX_500_POINTS_LOCAL_DATETIME, index=False)


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