import numpy as np
import pandas as pd
import geopandas as gp
import timezonefinder
import seaborn as sns

import statistics as st
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytz
import datetime_plots as dt
from contextlib import suppress
import os


from configuration import BASE_DIR, CHECKINS, SPOTS_1, CATEGORY_STRUCTURE, TIMEZONES, BRASIL_STATES, COUNTRIES, \
    CHECKINS_LOCAL_DATETIME, CHECKINS_7_CATEGORIES, CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED, US_STATES, CA_COUNTIES, \
    NY_COUNTIES, CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED_US, US_COUNTIES


def barplot(metrics, x_column, y_column, base_dir, file_name, title):
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    plt.figure()
    figure = sns.barplot(x=x_column, y=y_column, data=metrics).set_title(title)
    figure = figure.get_figure()
    figure.savefig(base_dir + file_name + ".png", bbox_inches='tight', dpi=400)

if __name__ == "__main__":

    df = pd.read_csv(CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED_US)

    df['local_datetime'] = pd.to_datetime(df['local_datetime'], format="%Y-%m-%d %H:%M:%S")

    # year
    total_month = {2009: 0, 2010: 0, 2011: 0}
    datetime_list = df['local_datetime'].tolist()

    for i in range(len(datetime_list)):
        year = datetime_list[i].year
        total_month[year] += 1

    print(total_month)
    months = list(total_month.keys())
    total = list(total_month.values())
    month_df = pd.DataFrame({'Year': months, 'Total': total})
    barplot(month_df, 'Year', 'Total', "", "year.png", "")
    print(df)

    # month
    total_month = {i: 0 for i in range(1, 13)}
    min_datetime = pd.Timestamp(year=2010, month=1, day=1)
    max_datetime = pd.Timestamp(year=2011, month=1, day=1)
    df = df[((df['local_datetime']< max_datetime) & (df['local_datetime']>= min_datetime))]
    datetime_list = df['local_datetime'].tolist()

    for i in range(len(datetime_list)):
        month = datetime_list[i].month
        total_month[month] += 1

    print(total_month)
    months = list(total_month.keys())
    total = list(total_month.values())
    month_df = pd.DataFrame({'Month': months, 'Total': total})
    barplot(month_df, 'Month', 'Total', "", "month_2011.png", "")
    print(df)