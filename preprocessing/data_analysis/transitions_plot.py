import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from configuration import USERS_10_MIL_MAX_500_POINTS_LOCAL_DATETIME, USERS_STEPS_8_CATEGORIES_10_MIL_MAX_500_POINTS_WITH_DETECTED_POIS_WITH_OSM_POIS_FILENAME
sns.set_theme()

def save_fig(dir, filename, fig):
    Path(dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(dir + filename + ".png",
                bbox_inches='tight',
                dpi=400)

def hour_frequency_plot(hour_frequency_dict, dir, title, week):
    total = []
    total_frequency = sum(hour_frequency_dict.values())
    #total_frequency = 1
    for day in hour_frequency_dict:
        total.append(hour_frequency_dict[day] / total_frequency)
    df = pd.DataFrame({'Category': list(hour_frequency_dict.keys()), 'Percentage': total})

    barplot(dir, 'Category', 'Percentage', df, "barplot_category_total_" + week + title,
                 "Percentage of events per category" + title)

def barplot(dir, x, y, df, filename, title, save=True):

    plt.figure()
    fig = sns.barplot(x=x, y=y, data=df)
    fig = fig.set_title(title).get_figure()
    plt.xticks(rotation=35)

def transictions_per_user(df, transitions_dict, count_dict):

    df = df.sort_values('datetime')
    categories_list = df['poi_resulting'].tolist()

    count_dict[categories_list[0]] += 1
    for i in range(1, len(categories_list)):

        from_category = categories_list[i-1]
        to_category = categories_list[i]

        transitions_dict[from_category][to_category] += 1



def transitions_per_state(df, transitions_dict):

    df_state_user = df.groupby('installation_id').apply(lambda e: transictions_per_user(df, transitions_dict, {}))


if __name__ == "__main__":

    df = pd.read_csv(USERS_STEPS_8_CATEGORIES_10_MIL_MAX_500_POINTS_WITH_DETECTED_POIS_WITH_OSM_POIS_FILENAME)
    poi_resulting = df['poi_resulting']
    df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S")

    poi_resulting_list = poi_resulting.tolist()

    for i in range(len(poi_resulting_list)):

        s = poi_resulting_list[i]
        s = s[0].upper() + s[1:]
        poi_resulting_list[i] = s
    df['poi_resulting'] = np.array(poi_resulting_list)
    categories = pd.Series(poi_resulting_list).unique().tolist()
    print(df['poi_resulting'].unique().tolist())
    transitions_dict = {i: {j:0 for j in categories} for i in categories}
    frequency_dict = {i: 0 for i in categories}

    df_state = df.groupby('state_name').apply(lambda e: transitions_per_state(e, copy.copy(transitions_dict)))

    hour_frequency_plot(hour_week_day_frequency_dict, "", " on week days", "8_categories_frequency_weekday")
    hour_frequency_plot(hour_weekend_frequency_dict, "", " on weekends", "8_categories_frequency_weekend")
    hour_frequency_plot(frequency_dict, "", "", "8_categories_frequency")