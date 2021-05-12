import numpy as np
import pandas as pd

from extractor.file_extractor import FileExtractor
from model.confusion_matrix import ConfusionMatrix

class UserStepDomain:

    def __init__(self):
        self.file_extractor = FileExtractor()

    def users_steps_from_csv(self, filename):
        users_steps = self.file_extractor.read_csv(filename)
        # linha,id,datetime,latitude,longitude,handset,operating_system
        users_steps['datetime'] = np.array(users_steps['reference_date'].tolist())
        users_steps = users_steps[['installation_id', 'datetime', 'latitude', 'longitude']]
        users_steps.columns = ['id', 'datetime', 'latitude', 'longitude']
        users_steps['index'] = np.array([i for i in range(len(users_steps))])
        users_steps['id'] = users_steps['id'].astype('int64')
        users_steps['datetime'] = pd.to_datetime(users_steps['datetime'], infer_datetime_format=True)
        print("Describe datetime: ", users_steps['datetime'].describe())
        return users_steps

    def user_pois_from_csv(self, filename):
        """ id, poi_type, latitude, longitude, work_time_events, home_time_events, inactive_interval_start,
        inactive_interval_end, inactive_applied_flag, inverted_routine_flag
        """
        df = self.file_extractor.read_csv(filename)
        # gdf = gpd.GeoDataFrame(
        #     df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

        return df

    def ground_truth_from_csv(self, filename):
        """ID Usuario, Latitude, Longitude, Classe"""
        poi_type_to_eng = {"Casa": "home", "Trabalho": "work", "Outro": "other", "Lazer": "other"}
        df = self.file_extractor.read_csv(filename)
        df.columns = ['id', 'latitude', 'longitude', 'poi_type']
        df['poi_type'] = df['poi_type'].apply(lambda e: poi_type_to_eng[e])

        return df

    def read_csv(self, filename):

        return self.file_extractor.read_csv(filename)

    # def geodesic_point_buffer(lati, lon, km):
    #     # Azimuthal equidistant projection
    #     aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    #     proj_wgs84 = pyproj.Proj(init='epsg:4326')
    #     project = partial(
    #         pyproj.transform,
    #         pyproj.Proj(aeqd_proj.format(lat=lati, lon=lon)),
    #         proj_wgs84)
    #     buf = Point(0, 0).buffer(km * 1000)  # distance in metres
    #     return transform(project, buf).exterior.coords[:]


