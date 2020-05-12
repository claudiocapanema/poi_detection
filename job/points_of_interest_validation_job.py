from foundation.abs_classes.job import Job
from domain.user_step_domain import UserStepDomain
from foundation.configuration.input import Input
from domain.detected_points_of_interest_validation_domain import DetectedPointOfInterestValidationDomain
from domain.identified_points_of_interest_validation_domain import IdentifiedPointOfInterestValidationDomain

class PointsOfInterestValidation:

    def __init__(self):
        self.user_step_domain = UserStepDomain()
        self.detected_poi_validation_domain = DetectedPointOfInterestValidationDomain()
        self.identified_poi_validation_domain = IdentifiedPointOfInterestValidationDomain()


    def start(self):

        ground_truth_filename = Input.get_instance().inputs['ground_truth']
        detected_pois_filename = Input.get_arg("poi_detection_filename")
        classified_pois_filename = Input.get_arg("poi_classification_filename")
        ground_truth = self.user_step_domain.ground_truth_from_csv(ground_truth_filename)
        classified_pois = self.user_step_domain.user_pois_from_csv(classified_pois_filename)
        detected_pois = self.user_step_domain.user_pois_from_csv(detected_pois_filename)

        # validate identified pois (poi class is not taken in consideration)
        self.identified_poi_validation_domain.identified_pois_validation(detected_pois, ground_truth)
        # validate classified pois
        self.detected_poi_validation_domain.users_pois_validation(classified_pois, ground_truth,
                                                                  "Classified Pois Validation")
        # validate detected pois
        self.detected_poi_validation_domain.users_pois_validation(detected_pois, ground_truth,
                                                                  "Detected Pois Validation")


