import numpy as np
import pandas as pd
import geopandas as gp
import pytz
import datetime as dt
import json
import requests

from configuration import BASE_DIR, CHECKINS, CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED


import json
import numpy as np
import pandas as pd
import ast
import statistics as st

def user_duration(user):

    if len(user) <= 1:
        return np.nan
    else:
        min_date = user['local_datetime'].min()
        max_date = user['local_datetime'].max()

        duration = int(((max_date-min_date).total_seconds()/3600)/len(user))

        return pd.DataFrame({'duration': [duration], 'total': [len(user)]})

def users_duration(country):

    country = country.groupby('userid').apply(lambda e: user_duration(e)).dropna()
    if len(country) <= 1:
        return pd.DataFrame({'mean_duration': [np.nan], 'median_duration': [np.nan], 'mean_total': [np.nan], 'median_total': [np.nan]})
    mean = country['duration'].mean()
    median = country['duration'].median()
    mean_total = country['total'].mean()
    median_total = country['total'].median()
    return pd.DataFrame({'mean_duration': [mean], 'median_duration': [median], 'mean_total': [mean_total], 'median_total': [median_total]})

if __name__ == "__main__":

    report = ""
    df = pd.read_csv(CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED)
    n = len(df)
    top_countries = df.groupby('country_name').apply(lambda e: pd.DataFrame({'total': [len(e)]})).reset_index().sort_values('total', ascending=False)[['country_name', 'total']]
    top_countries_users = \
    df.groupby('country_name').apply(lambda e: pd.DataFrame({'total_users': [len(e['userid'].unique().tolist())]})).reset_index().sort_values('total_users',
                                                                                                            ascending=False)[['country_name', 'total_users']]

    # top_countries_users_duration = \
    #     df.groupby('country_name').apply(
    #         lambda e: users_duration(e)).reset_index().sort_values(
    #         'median_total',
    #         ascending=False)[['country_name', 'mean_duration', 'median_duration', 'mean_total', 'median_total']]

    print(top_countries)
    print(top_countries_users)
    n_users = len(df['userid'].unique().tolist())
    categories = df['category'].unique().tolist()
    n_countries = len(df['country_name'].unique().tolist())

    print(df)
    report = """Número de linhas: {} \nNúmero de usuários: {}\nCategorias: {}\nNúmero de países: {}
    \nPaíses com mais eventos: {}
    \nPaíses com mais usuários: {}""".format(n, n_users, categories, n_countries, top_countries.head(10).reset_index(drop=True), top_countries_users.head(10).reset_index(drop=True))

    print(report)
    report = pd.DataFrame({'report': [report]})
    report.to_csv('statistics.txt', header=False, index=False, sep=' ')