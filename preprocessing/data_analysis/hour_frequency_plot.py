import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from configuration import USERS_10_MIL_MAX_500_POINTS_LOCAL_DATETIME


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
    df = pd.DataFrame({'Hour': list(hour_frequency_dict.keys()), 'Total': total})

    barplot(dir, 'Hour', 'Total', df, "barplot_hour_total_" + week + title,
                 "Percentage of events per hour (" + week + ")" + title)

def barplot(dir, x, y, df, filename, title):

    plt.figure()
    fig = sns.barplot(x=x, y=y, data=df).set_title(title).get_figure()

    save_fig(dir, filename, fig)

if __name__ == "__main__":

    df = pd.read_csv(USERS_10_MIL_MAX_500_POINTS_LOCAL_DATETIME)
    df['local_datetime'] = pd.to_datetime(df['local_datetime'], infer_datetime_format=True)
    hour_week_day_frequency_dict = {i: 0 for i in range(24)}
    hour_weekend_frequency_dict = {i: 0 for i in range(24)}
    local_datetime = df['local_datetime'].tolist()

    for date in local_datetime:
        if date.weekday() < 5:
            hour_week_day_frequency_dict[date.hour] += 1
        else:
            hour_weekend_frequency_dict[date.hour] += 1

    hour_frequency_plot(hour_week_day_frequency_dict, "", "weekday hour", "weekday")
    hour_frequency_plot(hour_weekend_frequency_dict, "", "weekend hour", "weekend")