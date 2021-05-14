import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from configuration import USERS_10_MIL_MAX_500_POINTS_LOCAL_DATETIME, USERS_STEPS_10_MIL_MAX_500_POINTS_WITH_DETECTED_POIS_WITH_OSM_POIS_FILENAME


def save_fig(dir, filename, fig):
    Path(dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(dir + filename + ".png",
                bbox_inches='tight',
                dpi=400)

def hour_frequency_plot(hour_frequency_dict, dir, title, week):
    total = []
    total_frequency = sum(hour_frequency_dict.values())
    total_frequency = 1
    for day in hour_frequency_dict:
        total.append(hour_frequency_dict[day] / total_frequency)
    df = pd.DataFrame({'Category': list(hour_frequency_dict.keys()), 'Total': total})

    barplot(dir, 'Category', 'Total', df, "barplot_category_total_" + week + title,
                 "Percentage of events per category (" + week + ")" + title)

def barplot(dir, x, y, df, filename, title):

    plt.figure()
    fig = sns.barplot(x=x, y=y, data=df).set_title(title).get_figure()

    save_fig(dir, filename, fig)

if __name__ == "__main__":

    df = pd.read_csv(USERS_STEPS_10_MIL_MAX_500_POINTS_WITH_DETECTED_POIS_WITH_OSM_POIS_FILENAME)
    poi_resulting = df['poi_resulting']
    df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)
    categories = poi_resulting.unique().tolist()
    poi_resulting = poi_resulting.tolist()
    hour_week_day_frequency_dict = {i: 0 for i in categories}
    hour_weekend_frequency_dict = {i: 0 for i in categories}
    frequency_dict = {i: 0 for i in categories}

    local_datetime = df['datetime'].tolist()

    for i in range(len(local_datetime)):
        date = local_datetime[i]
        if date.weekday() < 5:
            hour_week_day_frequency_dict[poi_resulting[i]] += 1
        else:
            hour_weekend_frequency_dict[poi_resulting[i]] += 1

        frequency_dict[poi_resulting[i]] += 1

    hour_frequency_plot(hour_week_day_frequency_dict, "", "Weekday frequency categories", "frequency_weekday")
    hour_frequency_plot(hour_weekend_frequency_dict, "", "Weekend frequency categories", "frequency_weekend")
    hour_frequency_plot(frequency_dict, "", "Frequency categories", "frequency")