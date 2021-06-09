import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from configuration import BASE_DIR, CHECKINS, CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED_US
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
    df = pd.DataFrame({'Category': list(hour_frequency_dict.keys()), 'Percentage of records': total})

    barplot(dir, 'Category', 'Percentage of records', df, "barplot_category_total_" + week + title,
                 "Percentage of records per category" + title)

def barplot(dir, x, y, df, filename, title, save=True):

    plt.figure()
    fig = sns.barplot(x=x, y=y, data=df, color='cornflowerblue')
    fig = fig.set_title(title).get_figure()
    plt.xticks(rotation=35)
    save_fig(dir, filename, fig)

if __name__ == "__main__":

    df = pd.read_csv(CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED_US)
    poi_resulting = df['category']
    print("categorias: ", poi_resulting.unique().tolist())
    df['local_datetime'] = pd.to_datetime(df['local_datetime'], infer_datetime_format=True)

    poi_resulting_list = poi_resulting.tolist()

    for i in range(len(poi_resulting_list)):

        s = poi_resulting_list[i]
        s = s[0].upper() + s[1:]
        poi_resulting_list[i] = s
    categories = pd.Series(poi_resulting_list).unique().tolist()
    print(df['category'].unique().tolist())
    hour_week_day_frequency_dict = {i: 0 for i in categories}
    hour_weekend_frequency_dict = {i: 0 for i in categories}
    frequency_dict = {i: 0 for i in categories}

    local_local_datetime = df['local_datetime'].tolist()

    for i in range(len(local_local_datetime)):
        date = local_local_datetime[i]
        if date.weekday() < 5:
            hour_week_day_frequency_dict[poi_resulting_list[i]] += 1
        else:
            hour_weekend_frequency_dict[poi_resulting_list[i]] += 1

        frequency_dict[poi_resulting_list[i]] += 1

    hour_frequency_plot(hour_week_day_frequency_dict, "", " on week days", "7_categories_frequency_weekday")
    hour_frequency_plot(hour_weekend_frequency_dict, "", " on weekends", "7_categories_frequency_weekend")
    hour_frequency_plot(frequency_dict, "", "", "7_categories_frequency")