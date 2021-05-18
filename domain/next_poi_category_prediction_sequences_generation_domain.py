import numpy as np
import pandas as pd

from loader.next_poi_category_prediction_sequences_generation_loader import NextPoiCategoryPredictionSequencesGenerationLoader
from extractor.file_extractor import FileExtractor


class NextPoiCategoryPredictionSequencesGenerationDomain:

    def __init__(self, dataset_name):
        self.sequences_generation_for_poi_categorization_sequential_baselines_loader = NextPoiCategoryPredictionSequencesGenerationLoader()
        self.file_extractor = FileExtractor()
        self.dataset_name = dataset_name

    def read_csv(self, filename, datetime_column=None):

        df = self.file_extractor.read_csv(filename)
        print("colunas")
        print(df.columns)
        print(df)
        if datetime_column is not None:
            df[datetime_column] = pd.to_datetime(df[datetime_column], infer_datetime_format=True)
        df = df.dropna()

        return df


    def _user_steps_to_int(self,
                           df,
                           category_column,
                           locationid_column,
                           datetime_column,
                           categories_to_int):

        df = df.sort_values(by=datetime_column)

        categories_names = df[category_column].tolist()
        datetime_list = df[datetime_column].tolist()
        categories_id = []
        for i in range(len(categories_names)):
            categories_id.append(categories_to_int[categories_names[i]])

        df['category_id'] = np.array(categories_id)

        user_sequence = []
        user_hours = []
        user_id = df['id'].tolist()
        user_categories_ids = categories_id
        days_types = []

        for i in range(len(df)):
            date = datetime_list[i]
            week_day = date.weekday()
            if week_day < 5:
                day_type = 0
                hour = date.hour
            else:
                day_type = 1
                hour = date.hour + 24

            #
            sequence = [categories_id[i], hour, day_type, user_id[i]]
            user_sequence.append(sequence)

        return pd.DataFrame({'id': user_id[0], 'sequence': [str(user_sequence)], 'categories': [str(categories_id)]})

    def generate_sequences(self, users_checkins, sequences_size, max_pois, userid_column,
                           category_column,
                           locationid_column,
                           datetime_column,
                           categories_to_int):

        #users_checkins = users_checkins.head(10000)
        #df = users_checkins.query(str(userid_column)+" == '"+str(user_id) + "'")
        df = users_checkins.groupby(userid_column).apply(lambda e:self._user_steps_to_int(e,
                                                                                          category_column,
                                                                                          locationid_column,
                                                                                          datetime_column,
                                                                                          categories_to_int))

        #df = self._flatten_df(df, userid_column)

        return df

    def _flatten_df(self, df, userid_column):

        indexes = df.index.values
        users_sequences = []
        users_ids = []
        for i in range(indexes.shape[0]):

            user_df = df.loc[indexes[i][0]]
            if user_df.shape[0] <=3:
                continue

            users_ids.append(i)
            user_df[userid_column] = pd.Series([i]*user_df.shape[0])

            # location/hour/userid
            users_sequences.append(user_df.to_numpy())

        df = pd.DataFrame({userid_column: users_ids, 'user_sequence': users_sequences})
        return df

    def sequences_to_csv(self, df, users_sequences_folder, dataset_name, categories_type):

        filename = users_sequences_folder + dataset_name + "_" + categories_type + "_sequences.csv"

        self.sequences_generation_for_poi_categorization_sequential_baselines_loader.sequences_to_csv(df, filename)




