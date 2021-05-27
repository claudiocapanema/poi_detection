import pandas as pd

from configuration import USERS_STEPS_8_CATEGORIES_10_MIL_MAX_500_POINTS_WITH_DETECTED_POIS_WITH_OSM_POIS_FILENAME, USERS_10_MIL_MAX_500_POINTS_LOCAL_DATETIME

if __name__ == "__main__":

    df_original = pd.read_csv(USERS_10_MIL_MAX_500_POINTS_LOCAL_DATETIME)
    print("Tamanho dataset original: ", df_original)

    df = pd.read_csv(USERS_STEPS_8_CATEGORIES_10_MIL_MAX_500_POINTS_WITH_DETECTED_POIS_WITH_OSM_POIS_FILENAME)
    print(df)

    events_per_country = df.groupby(by='country_name').apply(lambda e: pd.DataFrame({'Total events': [len(e)]})).reset_index()[['country_name', 'Total events']]
    print(events_per_country)

    events_per_category_per_country = df.groupby(by=['poi_resulting', 'country_name']).apply(lambda e: e)
    print(events_per_category_per_country)