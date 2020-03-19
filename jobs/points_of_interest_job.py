import pandas as pd

from domains.user_step_domain import UserStepDomain
from foundation.abs_classes.job import Job
from domains.points_of_interest_domain import PointsOfInterestDomain
from loaders.file_loader import FileLoader
from foundation.configuration.input import Input

class PointOfInterest(Job):

    def __init__(self):
        self.user_steps_domain = UserStepDomain()
        self.points_of_interest_domain = PointsOfInterestDomain()
        self.file_loader = FileLoader()

    def start(self):
        users_steps = self.user_steps_domain.users_steps_from_csv()
        filename = Input.get_arg("poi_detection_output")
        print("users steps read", Input.get_args(), "tamanho: ", users_steps.shape)

        users_pois_classified = users_steps.groupby(by='id').\
            apply(lambda e: self.points_of_interest_domain.
                  individual_point_interest(e['id'].tolist(), e['latitude'].tolist(), e['longitude'].tolist(), e['datetime'].tolist()))

        users_pois_classified_concatenated = pd.DataFrame({"id": [], "poi_type": [], "latitude": [], "longitude": [],
                           "work_time_events": [], "home_time_events": []})

        for i in range(users_pois_classified.shape[0]):
            users_pois_classified_concatenated = users_pois_classified_concatenated.\
                append(users_pois_classified.iloc[i], ignore_index=True)

        users_pois_classified_concatenated['id'] = users_pois_classified_concatenated['id'].astype('int64')
        print("filename: ", filename)
        self.file_loader.save_df_to_csv(users_pois_classified_concatenated, filename)

