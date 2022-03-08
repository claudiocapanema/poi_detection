import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from configuration import USERS_10_MIL_MAX_500_POINTS_LOCAL_DATETIME, USER_TRACKING_HOME_WORK_OTHER_AND_GOWALLA_CATEGORIES
sns.set_theme(style='whitegrid')

def save_fig(dir, filename, fig):
    Path(dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(dir + filename,
                bbox_inches='tight',
                dpi=400)

def hour_frequency_plot(hour_frequency_dict, dir, title, week):
    total = []
    total_frequency = sum(hour_frequency_dict.values())
    #total_frequency = 1
    for day in hour_frequency_dict:
        total.append(hour_frequency_dict[day]*100 / total_frequency)
    df = pd.DataFrame({'Category': list(hour_frequency_dict.keys()), 'Percentage': total})

    barplot(dir, 'Category', 'Percentage', df, "users_steps_barplot_category_total_" + week + title + ".png",
                 "Percentage of records per category" + title)

def barplot(dir, x, y, df, filename, title, save=True):

    #plt.figure()
    sns.set(font_scale=1.6, style='whitegrid')
    fig = plt.figure(figsize=(8, 4))
    fig = sns.barplot(x=y, y=x, data=df, color='cornflowerblue', order=['Home', 'Work', 'Shopping','Outdoors', 'Community', 'Other',  'Food', 'Travel', 'Entertainment', 'Nightlife'])
    fig.set_ylabel("")
    fig = fig.set_title(title).get_figure()
    #plt.xticks(rotation=35)
    save_fig(dir, filename, fig)
    save_fig(dir, filename.replace("png", "svg"), fig)

if __name__ == "__main__":

    df = pd.read_csv(USER_TRACKING_HOME_WORK_OTHER_AND_GOWALLA_CATEGORIES).query("poi_resulting in ['Home', 'Work', 'Other', 'Shopping', 'Community', 'Food', 'Entertainment', 'Travel', 'Outdoors', 'Nightlife']")
    poi_resulting = df['poi_resulting']
    print("categorias: ", poi_resulting.unique().tolist())
    df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)

    poi_resulting_list = poi_resulting.tolist()

    for i in range(len(poi_resulting_list)):

        s = poi_resulting_list[i]
        s = s[0].upper() + s[1:]
        poi_resulting_list[i] = s
    categories = pd.Series(poi_resulting_list).unique().tolist()
    print(df['poi_resulting'].unique().tolist())
    hour_week_day_frequency_dict = {i: 0 for i in categories}
    hour_weekend_frequency_dict = {i: 0 for i in categories}
    frequency_dict = {i: 0 for i in categories}

    local_datetime = df['datetime'].tolist()

    for i in range(len(local_datetime)):
        date = local_datetime[i]
        if date.weekday() < 5:
            hour_week_day_frequency_dict[poi_resulting_list[i]] += 1
        else:
            hour_weekend_frequency_dict[poi_resulting_list[i]] += 1

        frequency_dict[poi_resulting_list[i]] += 1

    hour_frequency_plot(hour_week_day_frequency_dict, "", " on week days", "10_categories_frequency_weekday")
    hour_frequency_plot(hour_weekend_frequency_dict, "", " on weekends", "10_categories_frequency_weekend")
    hour_frequency_plot(frequency_dict, "", "", "10_categories_frequency")