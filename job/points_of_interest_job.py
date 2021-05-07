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
        poi_detection_filename = Input.get_arg("poi_detection_filename")
        users_steps_join_detected_pois = Input.get_instance().inputs['users_steps_join_detected_pois']
        users_steps = self.user_step_domain.users_steps_from_csv(users_step_filename)
        users_steps = users_steps[users_steps.datetime < pd.Timestamp(year=2018, month=8, day=1)]
        users_steps = users_steps[users_steps.datetime >= pd.Timestamp(year=2018, month=7, day=1)]

        # ----------------------------------------

        """
        Detecting (Identifying and classifying together) PoIs of each user
        """
        #users_detected_pois = self.users_pois_detection(users_steps, utc_to_sp, poi_detection_filename)
        # ----------------------------------------

        """
        Classifying the PoIs of each user (the PoIs are given by the ground truth)
        """
        #self.users_pois_classificaion(users_steps, utc_to_sp)

        if users_steps_join_detected_pois == "yes":
            users_detected_pois = self.user_step_domain.read_csv(poi_detection_filename)
            users_detected_pois['id'] = users_detected_pois['id'].astype('int')
            self.points_of_interest_domain.associate_users_steps_with_pois(users_steps, users_detected_pois)

    def users_pois_detection(self, users_steps, utc_to_sp, poi_detection_filename):

        users_pois_detected = users_steps.groupby(by='id'). \
            apply(lambda e: self.points_of_interest_domain.
                  identify_points_of_interest(e,
                                              utc_to_sp))

        """
        Organizing the results into a single table
        """
        users_pois_detected_concatenated = self.points_of_interest_domain. \
            concatenate_dataframes(users_pois_detected)
        self.file_loader.save_df_to_csv(users_pois_detected_concatenated, poi_detection_filename)

        return users_pois_detected_concatenated

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



