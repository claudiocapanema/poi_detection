import pandas as pd
import geopandas as gpd

from foundation.configuration.input import Input

class FileExtractor:

    def __init__(self):
        self.users_steps_csv_filename = Input.get_instance().inputs['users_steps_filename']
        self.ground_truth_filename = Input.get_instance().inputs['ground_truth']

    def read_csv(self, filename):

        df = pd.read_csv(filename)

        return df

    def extract_ground_truth_from_csv(self):
        df = pd.read_csv(self.ground_truth_filename)

        return df