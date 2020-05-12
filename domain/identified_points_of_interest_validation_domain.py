import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import haversine_distances

from model.confusion_matrix import ConfusionMatrix
from configuration.detected_points_of_interest_validation_configuration import DetectedPointsOfInterestValidationConfiguration
from foundation.general_code.nearest_neighbors import NearestNeighbors

class IdentifiedPointOfInterestValidationDomain:

    def __init__(self):

        self.identification_confusion_matrix = ConfusionMatrix()

    def identified_pois_validation(self, identified_pois, ground_truth):
        print("------------------ \nIdentified Pois Validation")
        ids = ground_truth['id'].unique().tolist()
        ids_users_with_inverted_routine = []

        for i in ids:
            """
                Processing each user
            """
            query = """id=={}""".format(str(i))
            gt = ground_truth.query(query)

            ip = identified_pois.query(query)
            dp_indexes = [j for j in range(ip.shape[0])]

            """
                Pre-processing to find nearest points of each ground truth point
            """
            gt_latitudes = gt['latitude'].tolist()
            gt_longitudes = gt['longitude'].tolist()
            gt_points = np.radians([(long, lat) for long, lat in zip(gt_latitudes, gt_longitudes)])
            ip_latitudes = ip['latitude'].tolist()
            ip_longitudes = ip['longitude'].tolist()
            ip_points = np.radians([(long, lat) for long, lat in zip(ip_latitudes, ip_longitudes)])
            distances, indexes = NearestNeighbors.\
                find_radius_neighbors(gt_points, ip_points,
                                      DetectedPointsOfInterestValidationConfiguration.RADIUS.get_value())
            validated_indexes = []

            """
                Calculating the metrics
            """

            for j in range(len(indexes)):
                found_poi_flag = False

                """
                    Sorting nearest points by distance
                """
                result = [(dis, ind) for dis, ind in zip(distances[j], indexes[j])]
                result = sorted(result, key=lambda e:e[0])

                validated_indexes = []
                if len(result)>0:  # indexes
                    # if distances[i][j] > RADIUS or distances[i][j] * 6371 > 0.1:
                    #     print("erro: ", distances[i][j], " raio: ", RADIUS)
                    row = ip.iloc[result[0][1]]

                    """
                        it gets the users that have inverted routine, that is, 
                        the users that have night work
                    """

                    validated_indexes.append(result[0][1])
                    self._add_tp()
                else:
                    self._add_fn()

            new_dp_indexes = []
            for j in dp_indexes:
                if j not in validated_indexes:
                    new_dp_indexes.append(j)

            self._calculate_fp(ip, new_dp_indexes)

        self._classification_report()

    def _classification_report(self):
        self.identification_confusion_matrix.classification_report()

    def _calculate_fp(self, dp, new_dp_indexes):
        for i in new_dp_indexes:
            self._add_fp()

    def _add_tp(self):
        self.identification_confusion_matrix.add_tp()

    def _add_fp(self):
        self.identification_confusion_matrix.add_fp()

    def _add_fn(self):
        self.identification_confusion_matrix.add_fn()

    def _add_tn(self):
        self.identification_confusion_matrix.add_tn()

    def _set_total_samples_of_poi_type(self, total):
        self.identification_confusion_matrix.set_total_samples_of_poi_type(total)