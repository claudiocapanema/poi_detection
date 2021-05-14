import numpy as np
import pandas as pd
from configuration import USERS_10_MIL_MAX_500_POINTS, COUNTRIES, USERS_10_MIL_MAX_500_POINTS_LOCAL_DATETIME
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
    print("Tamanho inicial: ", len(df))
    gdf = gp.GeoDataFrame(
        df, geometry=gp.points_from_xy(df.longitude, df.latitude))

    gf = gp.read_file(COUNTRIES)
    gf = gp.sjoin(gf, gdf, op='contains')
    gf = gf[['id', 'installation_id', 'reference_date', 'latitude', 'longitude', 'CNTRY_NAME']]
    gf.columns = ['id', 'installation_id', 'reference_date', 'latitude', 'longitude', 'country_name']
    print("Tamanho final: ", len(gf))
    print(gf)

    print(gf['reference_date'])
    reference_date = gf['reference_date'].tolist()
    latitude = gf['latitude'].tolist()
    longitude = gf['longitude'].tolist()
    countries = gf['country_name'].tolist()
    print("Fora do brasil: ", len(gf.query("country_name != 'Brazil'")))
    print("rodar")

    local_datetimes = []
    p = int(len(gf)/10)
    n = 10
    for i in range(len(gf)):
        date = reference_date[i]
        if countries[i] == "Brazil":
            local_datetime = datetime_utils.convert_tz(date, pytz.utc, 'America/Sao_Paulo')
            local_datetimes.append(local_datetime)
        else:
            lat = latitude[i]
            lon = longitude[i]
            local_datetime = datetime_utils.find_timezone(date, lat, lon)
            local_datetimes.append(local_datetime)
        if i == p:
            p = p + p
            print("Roudou " + str(n) + "%")
            n = n + 10

    gf['local_datetime'] = np.array(local_datetimes)

    gf[['id', 'installation_id', 'local_datetime', 'reference_date', 'latitude', 'longitude', 'country_name']].to_csv(USERS_10_MIL_MAX_500_POINTS_LOCAL_DATETIME, index=False)