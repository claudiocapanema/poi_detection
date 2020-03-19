import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import haversine_distances

from models.confusion_matrix import ConfusionMatrix

class PointOfInterestValidationDomain:

    def __init__(self):

        self.home_confusion_matrix = ConfusionMatrix('home')
        self.work_confusion_matrix = ConfusionMatrix('work')
        self.other_confusion_matrix = ConfusionMatrix('other')

    def detected_pois_validation(self, detected_pois, ground_truth):
        RADIUS = 0.1 / 6371
        ids = ground_truth['id'].unique().tolist()
        ids_users_with_inverted_routine = []

        for i in ids:
            """
                Processing each user
            """
            query = """id=={}""".format(str(i))
            gt = ground_truth.query(query)

            dp = detected_pois.query(query)
            dp_indexes = [j for j in range(dp.shape[0])]

            """
                Pre-processing to find nearest points of each ground truth point
            """
            gt_latitudes = gt['latitude'].tolist()
            gt_longitudes = gt['longitude'].tolist()
            gt_points = np.radians([(long, lat) for long, lat in zip(gt_latitudes, gt_longitudes)])
            dp_latitudes = dp['latitude'].tolist()
            dp_longitudes = dp['longitude'].tolist()
            dp_points = np.radians([(long, lat) for long, lat in zip(dp_latitudes, dp_longitudes)])
            distances, indexes = self._find_nearest(gt_points, dp_points)
            validated_indexes = []

            """
                Calculating the metrics
            """

            for j in range(len(indexes)):
                poi_type = gt['poi_type'].iloc[j]
                found_poi_flag = False
                result = [(dis, ind) for dis, ind in zip(distances[j], indexes[j])]
                result = sorted(result, key=lambda e:e[0])
                validated_indexes = []
                for k in range(len(result)):  # indexes
                    # if distances[i][j] > RADIUS or distances[i][j] * 6371 > 0.1:
                    #     print("erro: ", distances[i][j], " raio: ", RADIUS)
                    if dp['poi_type'].iloc[result[k][1]] == poi_type:
                        row = dp.iloc[result[k][1]]

                        if row['inverted_routine_flag']:
                            print("flag: ", row['inverted_routine_flag'], row['id'])
                            if str(row['id']) not in ids_users_with_inverted_routine and poi_type != "other":
                                ids_users_with_inverted_routine.append(row['id'])
                            if poi_type == "home":
                                self.home_confusion_matrix.add_total_users_inverted_routine_tp()
                            elif poi_type == "work":
                                self.work_confusion_matrix.add_total_users_inverted_routine_tp()

                        validated_indexes.append(result[k][1])
                        self._add_tp(poi_type)
                        found_poi_flag = True
                        break
                if not found_poi_flag:
                    self._add_fp(poi_type)

            new_dp_indexes = []
            for j in dp_indexes:
                if j not in validated_indexes:
                    new_dp_indexes.append(j)

            self._calculate_fp(dp, new_dp_indexes)
            self._count_samples_of_each_poi_type(gt)

        number_users_inverted_routine_tp = pd.Series(ids_users_with_inverted_routine).astype('object').describe()
        self._classification_report(number_users_inverted_routine_tp)

    def _classification_report(self, number_users_inverted_routine_tp):
        print("Usuarios com rotina invertida tp: ", number_users_inverted_routine_tp)
        self.home_confusion_matrix.classification_report()
        self.work_confusion_matrix.classification_report()
        self.other_confusion_matrix.classification_report()

    def _count_samples_of_each_poi_type(self, gt):

        describe = gt.groupby(by='poi_type').count()
        try:
            total_home = describe.loc['home'].iloc[0]
            self._set_total_samples_of_poi_type(total_home, 'home')
        except Exception as e:
            pass

        try:
            total_work = describe.loc['work'].iloc[0]
            self._set_total_samples_of_poi_type(total_work, 'work')
        except Exception as e:
            pass

        try:
            total_other = describe.loc['other'].iloc[0]
            #print("Total other: ", total_other)
            self._set_total_samples_of_poi_type(total_other, 'other')
        except Exception as e:
            pass

    def _calculate_fp(self, dp, new_dp_indexes):
        for i in new_dp_indexes:
            poi_type = dp.iloc[i].loc['poi_type']
            self._add_fp(poi_type)

    def _find_nearest(self, gt_points, dp_points):
        RADIUS = 0.1 / 6371
        neigh = NearestNeighbors(radius=RADIUS, algorithm='ball_tree', metric='haversine')
        neigh = neigh.fit(dp_points)
        rng = neigh.radius_neighbors(gt_points)
        distances = rng[0]
        indexes = rng[1]
        return distances, indexes
        # if len(indexes[0]) > 0:
        #     n_distance = indexes[0][0]
        #     n_distance = np.degrees([n_distance])*6371
        #     print("Do sklearn: ", n_distance)
        #     original = haversine_distances([dp_points[0], gt_points[0]])[0][1]
        #     distancia = (original*6371)
        #     print("Distancia: ", "original: ", original, "depois: ", distancia )
        #     if distancia > 0.1:
        #         print("coord: ", np.degrees(gt_points[0]), np.degrees(dp_points[0]))

    def _add_tp(self, poi_type):
        if poi_type == "home":
            self.home_confusion_matrix.add_tp()
        elif poi_type == "work":
            self.work_confusion_matrix.add_tp()
        else:
            self.other_confusion_matrix.add_tp()

    def _add_fp(self, poi_type):
        if poi_type == "home":
            self.home_confusion_matrix.add_fp()
        elif poi_type == "work":
            self.work_confusion_matrix.add_fp()
        else:
            self.other_confusion_matrix.add_fp()

    def _add_fn(self, poi_type):
        if poi_type == "home":
            self.home_confusion_matrix.add_fn()
        elif poi_type == "work":
            self.work_confusion_matrix.add_fn()
        else:
            self.other_confusion_matrix.add_fn()

    def _add_tn(self, poi_type):
        if poi_type == "home":
            self.home_confusion_matrix.add_tn()
        elif poi_type == "work":
            self.work_confusion_matrix.add_tn()
        else:
            self.other_confusion_matrix.add_tn()

    def _set_total_samples_of_poi_type(self, total, poi_type):
        if poi_type == "home":
            self.home_confusion_matrix.set_total_samples_of_poi_type(total)
        elif poi_type == "work":
            self.work_confusion_matrix.set_total_samples_of_poi_type(total)
        else:
            self.other_confusion_matrix.set_total_samples_of_poi_type(total)