import numpy as np
import pandas as pd
import os
from contextlib import suppress

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
        users_detected_pois_with_osm_pois_filename = Input.get_instance().inputs['users_detected_pois_with_osm_pois_filename']
        users_steps_with_detected_pois_with_osm_pois_filename = Input.get_instance().inputs['users_steps_with_detected_pois_with_osm_pois_filename']
        users_steps = self.user_step_domain.users_steps_from_csv(users_step_filename)
        min_datetime = pd.Timestamp(year=2018, month=6, day=30)
        # max_datetime = pd.Timestamp(year=2018, month=9, day=1)
        # users_steps = users_steps[users_steps.datetime < max_datetime]
        users_steps = users_steps[users_steps.datetime >= min_datetime]
        #users_steps = self.select_article_users(users_steps)
        print("Filtrado")
        print(users_steps['datetime'].describe())
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
            poi_resulting_to_int = {'Tourism': 0, 'Amenity': 1, 'Leisure': 2, 'Shop': 3, 'Commuting': 4, 'Home': 5, 'Work': 6, 'Other': 7}
            with suppress(OSError):
                os.remove(users_steps_with_detected_pois_with_osm_pois_filename)
            users_detected_pois = self.user_step_domain.read_csv(users_detected_pois_with_osm_pois_filename)
            print("llaa")
            print(users_detected_pois)
            users_detected_pois['id'] = users_detected_pois['id'].astype('int')
            print("ca", users_detected_pois['poi_osm'].unique().tolist())
            users_steps['id'] = users_steps['id'].astype('int')
            users_steps_ids = users_steps['id'].unique().tolist()
            first_half = int(len(users_steps_ids)/7)
            second_half = first_half*2
            third_half = first_half*3
            fourth_half = first_half*4
            fifth_half = first_half*5
            sixth_half = first_half*6

            users_steps_with_pois = self.points_of_interest_domain.associate_users_steps_with_pois(users_steps.query("id in " + str(users_steps_ids[:first_half])),
                                                                                                   users_detected_pois.query("id in " + str(users_steps_ids[:first_half])))

            print("primeiros pois: ", users_steps_with_pois)
            #users_steps_with_pois = self.add_poi_resulting_id_column(users_steps_with_pois, poi_resulting_to_int)
            print("Salvar 1")
            print(users_steps_with_pois)
            self.file_loader.save_df_to_csv(users_steps_with_pois, users_steps_with_detected_pois_with_osm_pois_filename, 'a')

            users_steps_with_pois = self.points_of_interest_domain.associate_users_steps_with_pois(
                users_steps.query("id in " + str(users_steps_ids[first_half:second_half])), users_detected_pois.query("id in " + str(users_steps_ids[first_half:second_half])))
            #users_steps_with_pois = self.add_poi_resulting_id_column(users_steps_with_pois, poi_resulting_to_int)
            print("Salvar 2")
            print(users_steps_with_pois)
            self.file_loader.save_df_to_csv(users_steps_with_pois,
                                            users_steps_with_detected_pois_with_osm_pois_filename, 'a', False)

            users_steps_with_pois = self.points_of_interest_domain.associate_users_steps_with_pois(
                users_steps.query("id in " + str(users_steps_ids[second_half:third_half])),
                users_detected_pois.query("id in " + str(users_steps_ids[second_half:third_half])))
            #users_steps_with_pois = self.add_poi_resulting_id_column(users_steps_with_pois, poi_resulting_to_int)
            print("Salvar 3")
            print(users_steps_with_pois)
            self.file_loader.save_df_to_csv(users_steps_with_pois,
                                            users_steps_with_detected_pois_with_osm_pois_filename, 'a', False)

            users_steps_with_pois = self.points_of_interest_domain.associate_users_steps_with_pois(
                users_steps.query("id in " + str(users_steps_ids[third_half:fourth_half])),
                users_detected_pois.query("id in " + str(users_steps_ids[third_half:fourth_half])))
            #users_steps_with_pois = self.add_poi_resulting_id_column(users_steps_with_pois, poi_resulting_to_int)
            print("Salvar 4")
            print(users_steps_with_pois)
            self.file_loader.save_df_to_csv(users_steps_with_pois,
                                            users_steps_with_detected_pois_with_osm_pois_filename, 'a', False)

            users_steps_with_pois = self.points_of_interest_domain.associate_users_steps_with_pois(
                users_steps.query("id in " + str(users_steps_ids[fourth_half:fifth_half])),
                users_detected_pois.query("id in " + str(users_steps_ids[fourth_half:fifth_half])))
            #users_steps_with_pois = self.add_poi_resulting_id_column(users_steps_with_pois, poi_resulting_to_int)
            print("Salvar 5")
            print(users_steps_with_pois)
            self.file_loader.save_df_to_csv(users_steps_with_pois,
                                            users_steps_with_detected_pois_with_osm_pois_filename, 'a', False)

            users_steps_with_pois = self.points_of_interest_domain.associate_users_steps_with_pois(
                users_steps.query("id in " + str(users_steps_ids[fifth_half:sixth_half])),
                users_detected_pois.query("id in " + str(users_steps_ids[first_half:sixth_half])))
            #users_steps_with_pois = self.add_poi_resulting_id_column(users_steps_with_pois, poi_resulting_to_int)
            print("Salvar 6")
            print(users_steps_with_pois)
            self.file_loader.save_df_to_csv(users_steps_with_pois,
                                            users_steps_with_detected_pois_with_osm_pois_filename, 'a', False)

            users_steps_with_pois = self.points_of_interest_domain.associate_users_steps_with_pois(
                users_steps.query("id in " + str(users_steps_ids[sixth_half:])),
                users_detected_pois.query("id in " + str(users_steps_ids[sixth_half:])))
            #users_steps_with_pois = self.add_poi_resulting_id_column(users_steps_with_pois, poi_resulting_to_int)
            print("Salvar 6")
            print(users_steps_with_pois)
            self.file_loader.save_df_to_csv(users_steps_with_pois,
                                            users_steps_with_detected_pois_with_osm_pois_filename, 'a', False)

    def add_poi_resulting_id_column(self, users_steps_with_pois, poi_resulting_to_int):

        poi_category_id = []
        poi_resulting = users_steps_with_pois['poi_resulting'].tolist()
        for i in range(len(poi_resulting)):
            # print("indice", i)
            # print("categoria: ", poi_resulting[i])
            # print("inteiro: ", poi_resulting_to_int[poi_resulting[i]])
            poi_category_id.append(poi_resulting_to_int[poi_resulting[i]])

        users_steps_with_pois['poi_resulting_id'] = np.array(poi_category_id)

        return users_steps_with_pois

    def users_pois_detection(self, users_steps, utc_to_sp, poi_detection_filename):

        poi_id_count = 0
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

    def select_article_users(self, users_steps):

        filename = "/media/claudio/Data/backup_linux/Documentos/users_steps_datasets/df_mais_de_5_mil_limite_500_pontos.csv"
        users_ids = pd.read_csv(filename)['installation_id'].unique().tolist()

        users_steps = users_steps.query("id in {}".format(str(users_ids)))

        print("Quantidade de usuários selecionados: ", len(users_steps['id'].unique().tolist()))

        return users_steps


