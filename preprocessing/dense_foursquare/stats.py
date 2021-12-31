import pandas as pd
from configurations import BASE_DIR, DENSE_FOURSQUARE_LOCAL_DATETIME_US_BRAZIL_STATES_9_CATEGORIES
import pandas as np


def stat(state):

    count_events = len(state)
    count_users = len(state['userid'].unique().tolist())
    data_min = state['local_datetime'].min()
    data_max = state['local_datetime'].max()
    duration = data_max - data_min
    max_state = state.groupby('state_name').apply(lambda e: pd.DataFrame({'count_state': [len(e['userid'].unique().tolist())]})).sort_values('count_state', ascending=False).reset_index().head(1)
    max_state_count = max_state['count_state']
    max_state = max_state['state_name']

    state = pd.DataFrame({'count_events': [count_events], 'count_users': [count_users], 'data_min': [data_min],
                          'data_max': [data_max], 'duration': [duration], 'max_state': max_state, 'max_state_count': max_state_count})

    return state

def filtro_min_events_estado(df, report):

    min_events = 15

    df = df.query("country_name == 'Brazil' or country_name == 'United States'")

    df = df.query("state_name == 'CALIFORNIA' or state_name == 'São Paulo'")

    brazil = df.query("country_name == 'Brazil'")
    us = df.query("country_name == 'United States'")

    brazil_users = brazil.groupby('userid').apply(lambda e: pd.DataFrame({'count_events': [len(e)]}) if len(e) > min_events else pd.DataFrame({'count_events': [-1]})).query("count_events != -1")
    report += """\nUsuarios brasileiros com mais de {} eventos no estado de São Paulo: {}""".format(min_events, len(brazil_users))

    us_users = us.groupby('userid').apply(
        lambda e: pd.DataFrame({'count_events': [len(e)]}) if len(e) > min_events else pd.DataFrame(
            {'count_events': [-1]})).query("count_events != -1")
    report += """\nUsuarios americanos com mais de {} eventos no estado da California: {}""".format(min_events, len(us_users))

    return report

def filtro_min_events_pais(df, report):

    min_events = 15

    df = df.query("country_name == 'Brazil' or country_name == 'United States'")

    brazil = df.query("country_name == 'Brazil'")
    us = df.query("country_name == 'United States'")

    brazil_users = brazil.groupby('userid').apply(lambda e: pd.DataFrame({'count_events': [len(e)]}) if len(e) > min_events else pd.DataFrame({'count_events': [-1]})).query("count_events != -1")
    report += """\nUsuarios brasileiros com mais de {} eventos: {}""".format(min_events, len(brazil_users))

    us_users = us.groupby('userid').apply(
        lambda e: pd.DataFrame({'count_events': [len(e)]}) if len(e) > min_events else pd.DataFrame(
            {'count_events': [-1]})).query("count_events != -1")
    report += """\nUsuarios americanos com mais de {} eventos: {}""".format(min_events, len(us_users))

    return report


if __name__ == "__main__":

    report = ""
    df = pd.read_csv(DENSE_FOURSQUARE_LOCAL_DATETIME_US_BRAZIL_STATES_9_CATEGORIES)
    df['local_datetime'] = pd.to_datetime(df['local_datetime'], infer_datetime_format=True)
    df = df[['userid', 'local_datetime', 'country_name', 'state_name']]

    print(df)
    print(df.groupby('country_name').apply(lambda e: stat(e)).sort_values(by='count_users', ascending=False).reset_index())

    report = filtro_min_events_pais(df, report)
    report = filtro_min_events_estado(df, report)

    pd.DataFrame({'Relatório': [report]}).to_csv("relatório.csv", index_label=False, index=False)