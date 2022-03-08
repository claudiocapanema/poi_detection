import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

from configuration import BASE_DIR, CHECKINS, CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED_US
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

    barplot(dir, 'Category', 'Percentage', df, "gowalla_barplot_category_total_" + week + title + ".png",
                 "Percentage of records per category" + title)

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

def normaliseCounts(widths,maxwidth):
    widths = np.array(widths)/float(maxwidth)
    return widths

def barplot(dir, x, y, df, filename, title, save=True):

    sns.set(font_scale=1.6, style='whitegrid')

    #plt.figure()
    fig = plt.figure(figsize=(8, 4))
    #ax2 = ax.twinx()
    fig = sns.barplot(x=y, y=x, data=df, color='cornflowerblue',
                      order=['Food', 'Shopping', 'Community', 'Travel', 'Entertainment', 'Outdoors', 'Nightlife'])

    # widthbars = [1,1,1,1,1,1,1]
    #
    # widthbars = normaliseCounts(widthbars, 100)
    #
    # for bar, newwidth in zip(ax.patches, widthbars):
    #     x = bar.get_x()
    #     width = bar.get_height()
    #     centre = x + width / 2.
    #
    #     bar.set_y(centre - newwidth / 2.)
    #     bar.set_height(newwidth)
    #
    fig.set_ylabel("")
    #change_width(ax, .15)
    fig = fig.set_title(title).get_figure()

    #plt.xticks(rotation=35)

    save_fig(dir, filename, fig)
    save_fig(dir, filename.replace("png", "svg"), fig)

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