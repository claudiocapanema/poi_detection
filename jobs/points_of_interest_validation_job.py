from foundation.abs_classes.job import Job
from domains.user_step_domain import UserStepDomain
from foundation.configuration.input import Input
from domains.points_of_interest_validation_domain import PointOfInterestValidationDomain

class PointsOfInterestValidation:

    def __init__(self):
        self.user_step_domain = UserStepDomain()
        self.poi_validation_domain = PointOfInterestValidationDomain()


    def start(self):

        ground_truth = self.user_step_domain.ground_truth_from_csv()
        detected_pois = self.user_step_domain.detected_pois_from_csv()
        self.poi_validation_domain.detected_pois_validation(detected_pois, ground_truth)

