import pandas as pd

from foundation.configuration.data_sources import DataSources

class FileLoader:

    def __init__(self):
        pass

    def save_df_to_csv(self, df, filename):
        #filename = DataSources.FILES_DIRECTORY.get_value() + filename
        df.to_csv(filename)