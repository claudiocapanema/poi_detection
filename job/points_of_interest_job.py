import pandas as pd

from domain.user_step_domain import UserStepDomain
from foundation.abs_classes.job import Job
from domain.points_of_interest_domain import PointsOfInterestDomain
from loader.file_loader import FileLoader
from foundation.configuration.input import Input

class PointOfInterest(Job):

    def __init__(self):
        self.user_step_domain = UserStepDomain()
        self.points_of_interest_domain = PointsOfInterestDomain()
        self.file_loader = FileLoader()

    def start(self):
        users_step_filename = Input.get_instance().inputs['users_steps_filename']
        utc_to_sp = Input.get_instance().inputs['utc_to_sp']
        users_steps = self.user_step_domain.users_steps_from_csv(users_step_filename)

        # ----------------------------------------

        """
        Detecting (Identifying and classifying together) PoIs of each user
        """
        self.users_pois_detection(users_steps, utc_to_sp)
        # ----------------------------------------

        """
        Classifying the PoIs of each user (the PoIs are given by the ground truth)
        """
        self.users_pois_classificaion(users_steps, utc_to_sp)

    def users_pois_detection(self, users_steps, utc_to_sp):
        poi_detection_filename = Input.get_arg("poi_detection_filename")

        users_pois_detected = users_steps.groupby(by='id'). \
            apply(lambda e: self.points_of_interest_domain.
                  identify_points_of_interest(e['id'].tolist(), e['latitude'].tolist(),
                                              e['longitude'].tolist(), e['datetime'].tolist(),
                                              utc_to_sp))

        """
        Organizing the results into a single table
        """
        users_pois_detected_concatenated = self.points_of_interest_domain. \
            concatenate_dataframes(users_pois_detected)
        self.file_loader.save_df_to_csv(users_pois_detected_concatenated, poi_detection_filename)

    def users_pois_classificaion(self, users_steps, utc_to_sp):
        ground_truth_filename = Input.get_instance().inputs['ground_truth']
        poi_classification_filename = Input.get_arg("poi_classification_filename")

        ground_truth = self.user_step_domain.ground_truth_from_csv(ground_truth_filename)
        users_pois_classified = self.points_of_interest_domain.classify_pois_from_ground_truth(
            users_steps, ground_truth, utc_to_sp)
        """
        Organizing the results into a single table
        """
        users_pois_classified_concatenated = self.points_of_interest_domain. \
            concatenate_dataframes(users_pois_classified)
        self.file_loader.save_df_to_csv(users_pois_classified_concatenated, poi_classification_filename)


