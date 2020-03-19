import datetime as dt

def date_from_str_to_datetime(date):
    pattern = '%Y-%m-%d %H:%M:%S'
    return dt.datetime.strptime(date, pattern)