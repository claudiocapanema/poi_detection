import datetime as dt
import pytz

class DatetimesUtils:

    @classmethod
    def date_from_str_to_datetime(date):
        pattern = '%Y-%m-%d %H:%M:%S'
        return dt.datetime.strptime(date, pattern)

    @classmethod
    def convert_tz(cls, datetime, from_tz, to_tz):
        datetime = datetime.replace(tzinfo=from_tz)
        datetime = datetime.astimezone(pytz.timezone(to_tz))
        return  datetime