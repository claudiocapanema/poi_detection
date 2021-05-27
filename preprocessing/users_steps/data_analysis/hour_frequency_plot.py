import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from configuration import USERS_STEPS_8_CATEGORIES_10_MIL_MAX_500_POINTS_WITH_DETECTED_POIS_WITH_OSM_POIS_FILENAME
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
    df = pd.DataFrame({'Hour': list(hour_frequency_dict.keys()), 'Percentage of records': total})

    barplot(dir, 'Hour', 'Percentage of records', df, "barplot_hour_percentage_" + week + title,
                 "Percentage of records per hour on " + week)

def barplot(dir, x, y, df, filename, title):

    plt.figure()
    fig = sns.barplot(x=x, y=y, data=df, color='cornflowerblue').set_title(title).get_figure()
    plt.xticks(rotation=35)
    save_fig(dir, filename, fig)

def average_events_per_day(user):

    datetime = user['datetime']
    minimum = datetime.min()
    maximum = datetime.max()
    days = int((maximum - minimum).total_seconds()/86400)
    if days < 1:
        days = 1
    avg_events_per_day = int(len(datetime)/days)

    avg_events_per_day = pd.DataFrame({'events_per_day': [avg_events_per_day]})
    return avg_events_per_day

if __name__ == "__main__":

    df = pd.read_csv(USERS_STEPS_8_CATEGORIES_10_MIL_MAX_500_POINTS_WITH_DETECTED_POIS_WITH_OSM_POIS_FILENAME)
    df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S")
    hour_week_day_frequency_dict = {i: 0 for i in range(24)}
    hour_weekend_frequency_dict = {i: 0 for i in range(24)}

    print(df)
    average_users_steps_per_user_per_day = df.groupby('id').apply(lambda e: average_events_per_day(e))
    print("Evento por dia por usuário")
    print(average_users_steps_per_user_per_day['events_per_day'].describe())
    datetime = df['datetime'].tolist()
    for date in datetime:
        if date.weekday() < 5:
            hour_week_day_frequency_dict[date.hour] += 1
        else:
            hour_weekend_frequency_dict[date.hour] += 1

    hour_frequency_plot(hour_week_day_frequency_dict, "", "weekday hour", "week days")
    hour_frequency_plot(hour_weekend_frequency_dict, "", "weekend hour", "weekends")