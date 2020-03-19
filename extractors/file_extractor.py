import pandas as pd
import geopandas as gpd

from foundation.configuration.input import Input
from foundation.configuration.data_sources import DataSources

class FileExtractor:

    def __init__(self):
        self.users_steps_csv_filename = Input.get_instance().inputs['users_steps_csv']
        self.ground_truth_filename = Input.get_instance().inputs['ground_truth']
        self.detected_pois_filename = Input.get_instance().inputs['poi_detection_output']

    def extract_user_steps_from_csv(self):
        df = pd.read_csv(self.users_steps_csv_filename)
        return df

    def extract_detected_pois(self):

        df = pd.read_csv(self.detected_pois_filename)

        return df

    def extract_ground_truth_from_csv(self):
        df = pd.read_csv(self.ground_truth_filename)

        return df