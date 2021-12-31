import numpy as np
import pandas as pd
import geopandas as gdp

from configurations import DENSE_FOURSQUARE_LOCAL_DATETIME_9_CATEGORIES, BRAZIL_STATES_SHP, US_STATES_SHP, DENSE_FOURSQUARE_LOCAL_DATETIME_US_BRAZIL_STATES_9_CATEGORIES

if __name__ == "__main__":

    df = pd.read_csv(DENSE_FOURSQUARE_LOCAL_DATETIME_9_CATEGORIES)
    original_columns = list(df.columns)
    df = gdp.GeoDataFrame(
    df, geometry=gdp.points_from_xy(df.longitude, df.latitude))
    us_states = gdp.read_file(US_STATES_SHP).set_crs(4326, allow_override=True)[['State_Name', 'geometry']]
    us_states['state_name'] = np.array(us_states['State_Name'].tolist())
    us_states = us_states[['state_name', 'geometry']]
    brazil_states = gdp.read_file(BRAZIL_STATES_SHP).set_crs(4326, allow_override=True)[['nome', 'geometry']]
    brazil_states['state_name'] = np.array(brazil_states['nome'].tolist())
    brazil_states = brazil_states[['state_name', 'geometry']]
    states = brazil_states.append(us_states, ignore_index=True)
    df_us_states = gdp.sjoin(states, df, predicate='contains')[original_columns + ['state_name']]
    print(df)
    print(us_states.columns)
    print(brazil_states.columns)
    print(df_us_states)

    df_us_states.to_csv(DENSE_FOURSQUARE_LOCAL_DATETIME_US_BRAZIL_STATES_9_CATEGORIES, index=False)