import pandas as pd
import geopandas as gp
import folium
import matplotlib.pyplot as plt
from folium.plugins import HeatMap

from shapely.geometry import Point

def gerar_heatmap(jd):
    frac = 1
    country = "br_sp_sp"
    """ US tem 307136 coordenadas
        JP tem 203214 coordenadas
        BR tem 282778 coordenadas """
    # jd = jd.query("country_code in ['US']").sample(frac=frac)
    # print("estados us")
    # print(jd['state'].unique().tolist())
    # jd = jd.query("state == 'NY'")
    # jd = jd.query("county in ['New York', 'Bronx', 'Richmond', 'Queens', 'Kings', 'Manhattan', 'Brooklyn', 'Staten Island']")
    jd = jd.query("country_code in ['BR']").sample(frac=frac)
    print(jd['county'].unique().tolist())
    jd = jd.query("state == 'sp'")
    print("estados ")

    # print(jd)
    # print("cOUNTIES", jd['county'].unique().tolist())
    jd = jd.query(
        "county in ['SÃO PAULO']")

    print(jd.shape)

    lat = jd['latitude'].tolist()
    lng = jd['longitude'].tolist()

    m = folium.Map(location=[lat[0], lng[0]], zoom_start=2, tiles="Stamen Terrain")

    tooltip = "Click me!"

    HeatMap(list(zip(lat, lng))).add_to(m)

    m.save("countries_map_fracao_" + str(frac) + "_pais_" + country + ".html")

def statistics_br(df):
    print(df.columns)


    # total_samples = join.shape[0]
    # selected_states = join.query("country_code == ")

    df = df.query("country_code == 'BR'")
    total = len(df)
    total_users = len(df['userid'].unique().tolist())
    df_state = df.groupby(by='state').apply(lambda e: round(len(e)/total, ndigits=3)).reset_index()
    df_state.columns = ['state', 'percentage_records']
    df_state = df_state.sort_values(by='percentage_records', ascending=False)
    print(df_state)
    df_state_county = df.groupby(by=['state', 'county']).apply(lambda e: pd.DataFrame({'percentage_records': [round(len(e) / total, ndigits=3)],
                                                                                   'users': [len(e['userid'].unique().tolist())],
                                                                                  'percentage_users': [len(e['userid'].unique().tolist())/total_users]})).reset_index()
    #df_state_county.columns = ['state', 'county', , , ]
    df_state_county = df_state_county.sort_values(by='percentage_records', ascending=False)
    print(df_state_county)

def statistics_us(df):


    # total_samples = join.shape[0]
    # selected_states = join.query("country_code == ")

    df = df.query("country_code == 'US'")
    total = len(df)
    total_users = len(df['userid'].unique().tolist())
    df_state = df.groupby(by='state').apply(lambda e: round(len(e)/total, ndigits=3)).reset_index()
    df_state.columns = ['state', 'percentage']
    df_state = df_state.sort_values(by='percentage', ascending=False)
    print(df_state)
    df_state_county = df.groupby(by=['state', 'county']).apply(
        lambda e: pd.DataFrame({'percentage_records': [round(len(e) / total, ndigits=3)],
                                'users': [len(e['userid'].unique().tolist())],
                                'percentage_users': [len(e['userid'].unique().tolist()) / total_users]})).reset_index()
    # df_state_county.columns = ['state', 'county', , , ]
    df_state_county = df_state_county.sort_values(by='percentage_records', ascending=False)
    print(df_state_county)


if __name__ == "__main__":

    filename = "/media/claudio/Data/backup_win_hd/Downloads/doutorado/global_foursquare/dataset_TIST2015/dataset_TIST2015_Checkins_with_Pois_8_categories_local_datetime_br_us_jp_2012_2013_with_state_and_cities.csv"
    new = "/media/claudio/Data/backup_win_hd/Downloads/doutorado/global_foursquare/dataset_TIST2015/dataset_TIST2015_Checkins_with_Pois_8_categories_local_datetime_br_us_ca_ny_jp_2012_2013_with_state_and_cities.csv"
    jd = pd.read_csv(new).sample(frac=1)
    print(jd)
    gerar_heatmap(jd)

    statistics_br(jd)
    statistics_us(jd)


