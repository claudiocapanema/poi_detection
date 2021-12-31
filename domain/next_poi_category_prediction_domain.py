import copy
import statistics as st
import datetime as dt
import math

import numpy as np
import json
from scipy.stats import entropy

import pandas as pd
import sklearn.metrics as skm
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import utils as np_utils
from sklearn.model_selection import KFold
from spektral.layers.convolutional import GCNConv, ARMAConv, DiffusionConv

from extractor.file_extractor import FileExtractor
from foundation.util.next_poi_category_prediction_util import sequence_to_x_y, \
    sequence_tuples_to_spatial_temporal_and_feature8_ndarrays, \
    remove_hour_from_sequence_y
from foundation.util.nn_preprocessing import one_hot_decoding

from model.next_poi_category_prediction_models.users_steps.serm.model import SERMUsersSteps
from model.next_poi_category_prediction_models.users_steps.map.model import MAPUsersSteps
from model.next_poi_category_prediction_models.users_steps.stf.model import STFUsersSteps
from model.next_poi_category_prediction_models.users_steps.mfarnnuserssteps import MFARNNUsersSteps
from model.next_poi_category_prediction_models.users_steps.next.model import NEXTUsersSteps
from model.next_poi_category_prediction_models.gowalla.serm.serm import SERM
from model.next_poi_category_prediction_models.gowalla.map.map import MAP
from model.next_poi_category_prediction_models.gowalla.stf.stf import STF
from model.next_poi_category_prediction_models.gowalla.poi_rgnn.mfa_rnn import MFA_RNN
from model.next_poi_category_prediction_models.gowalla.next.next import NEXT
from model.next_poi_category_prediction_models.gowalla.garg.garg import GARG
from model.next_poi_category_prediction_models.users_steps.garg.garg import GARGUsersSteps
from model.next_poi_category_prediction_models.gowalla.poi_rgnne.poi_rgnne import POI_RGNNE

from loader.next_poi_category_prediction_loader import NextPoiCategoryPredictionLoader


class NextPoiCategoryPredictionDomain:


    def __init__(self, dataset_name, distance_sigma, duration_sigma):
        self.file_extractor = FileExtractor()
        self.next_poi_category_prediction_loader = NextPoiCategoryPredictionLoader()
        self.dataset_name = dataset_name
        self.distance_sigma = distance_sigma
        self.duration_sigma = duration_sigma
        self.count=0

    def _sequence_to_list(self, series):
        #print("original: ", series, type(series))
        #print("replace: ", series.replace("\n", "").replace(" ", ",").replace(".","").replace(",,",",").replace("[,", "[").replace(",,",",").replace("[,", "[").replace(",,", ","))
        if "." in series:
            self.count+=1
        #print("resul: ", series.replace("\n", "").replace(" ", ",").replace(".","").replace(",,",",").replace("[,", "[").replace(",,",","))
        #series = json.loads(series.replace("\n", "").replace(" ", ",").replace(".","").replace(",,",",").replace("[,", "[").replace(",,",",").replace("[,", "[").replace(",,", ","))
        series = json.loads(series.replace("'", ""))
        # series = series.replace("(", "[").replace(")", "]")
        # series = series.split("],")
        # new_series = []
        # for e in series:
        #     e = e.replace("[", "").replace("]", "")
        #     data, hour, location = e.split(",")
        #     new_series.append([data.replace(" '", "").replace("'", ""), int(hour.replace(" ", "")), int(location.replace(" ", ""))])

        # new_series = np.array(new_series)
        return np.array(series)

    def _add_total(self, user):

        total = []
        user = user.tolist()
        for i in range(len(user)):
            total.append(len(user[i]))

        return np.array(total)

    def read_sequences(self, filename, n_splits, model_name, number_of_categories, step_size, dataset_name):
        # 7000

        max_size = 4000
        df = self.file_extractor.read_csv(filename)
        df['sequence'] = df['sequence'].apply(lambda e: self._sequence_to_list(e))
        df['total'] = self._add_total(df['sequence'])
        df = df.sort_values(by='total', ascending=False)
        print(df['total'].describe())

        if dataset_name == "gowalla":
            minimum = 40
            n = 600
            #n = 100
            #minimum = 40
            #n = 1500
            #bom
            random = 1
        else:
            n = 1650
            minimum = 300
            #n = 1300

            #razoavel
            # minimum = 200
            # n = 1300

            # melhor
            #n = 1050
            #minimum = 250
            # melhor ainda
            # n = 1050
#            minimum = 300
            #random = 2
            # melhor 2
            #n = 1450
            #minimum = 300
            # melhor 3
            #n = 1650
            #minimum = 300
            #random = 4
            random = 4

        df = df.query("total >= " + str(minimum))
        print("usuarios com mais de " + str(minimum), len(df))
        df = df.sample(n=n, random_state=random)
        print(df)

        # reindex ids
        df['id'] = np.array([i for i in range(len(df))])

        users_ids = df['id'].tolist()
        sequences = df['sequence'].tolist()
        categories_list = df['categories'].tolist()
        x_list = []
        y_list = []
        countries = {}
        max_country = 0
        max_distance = 0
        max_duration = 0
        ids_remove = []

        distance_list = []
        duration_list = []

        maior_mes = 0
        for i in range(len(users_ids)):

            user_id = users_ids[i]
            sequence = sequences[i]
            categories = json.loads(categories_list[i])
            new_sequence = []

            if len(sequence) < minimum:
                x_list.append([])
                continue
            # # verify if user visited all categories
            # categories_size = len(categories)
            # part = int(categories_size/5)
            # first_index = part
            # second_index = part*2
            # third_index = part*3
            # fourth_index = part*4
            # fifth_index = categories_size
            # first = pd.Series(categories[:first_index]).unique().tolist()
            # second = pd.Series(categories[first_index:second_index]).unique().tolist()
            # third = pd.Series(categories[second_index:third_index]).unique().tolist()
            # fourth = pd.Series(categories[third_index:fourth_index]).unique().tolist()
            # fifth = pd.Series(categories[fourth_index:fifth_index]).unique().tolist()
            # if len(first) != number_of_categories or len(second) != number_of_categories or len(third) != number_of_categories \
            #         or len(fourth) != number_of_categories or len(fifth) != number_of_categories:
            #     x_list.append([])
            #     y_list.append([])
            #     continue

            size = len(sequence)
            for j in range(size):
                location_category_id = sequence[j][0]
                hour = sequence[j][1]
                if model_name in ['map', 'stf'] and hour >= 24:
                    hour = hour - 24
                country = sequence[j][2]
                distance = sequence[j][3]
                duration = sequence[j][4]
                #categories_distances_matrix = sequence[j][7]
                #categories_distances_matrix = self.categories_distances_matrix_preprocessing(categories_distances_matrix)
                if j < len(sequence) -1:
                    if duration > 72 and sequence[j+1][4] > 72:
                        continue
                week_day = sequence[j][5]
                poi_id = sequence[j][7]
                month = sequence[j][8]
                if month > maior_mes:
                    maior_mes = month

                if distance > 50:
                    distance = 50
                if duration > 72:
                    duration = 72
                distance_list.append(distance)
                duration_list.append(duration)
                countries[country] = 0
                if country > max_country:
                    max_country = country
                if distance > max_distance:
                    max_distance = distance
                if duration > max_duration:
                    max_duration = duration
                distance = self._distance_importance(distance)
                duration = self._duration_importance(duration)
                new_sequence.append([location_category_id, hour, country, distance, duration, week_day, user_id, poi_id, month])

            if dataset_name == "gowalla":
                x, y = sequence_to_x_y(new_sequence, step_size)
            elif dataset_name == "users_steps":
                x, y = sequence_to_x_y(new_sequence, step_size)
            y = remove_hour_from_sequence_y(y)

            user_df = pd.DataFrame({'x': x, 'y': y}).sample(frac=1, random_state=random)
            x = user_df['x'].tolist()
            y = user_df['y'].tolist()

            x_list.append(x)
            y_list.append(y)

        print("quantidade usuarios: ", len(users_ids))
        print("quantidade se: ", len(x_list))
        print("maior pais: ", max_country)
        print("maior distancia: ", max_distance)
        print("maior duracao: ", max_duration)
        print("distancia mediana: ", st.median(distance_list))
        print("duracao mediana: ", st.median(duration_list))
        print("maior mes: ", maior_mes)
        df['x'] = np.array(x_list)
        df['y'] = np.array(y_list)
        df = df[['id', 'x', 'y']]

        print("paises: ", len(list(countries.keys())))

        # remove users that have few samples
        ids_remove_users = []
        ids_ = df['id'].tolist()
        x_list = df['x'].tolist()
        #x_list = [json.loads(x_list[i]) for i in range(len(x_list))]
        for i in range(df.shape[0]):
            user = x_list[i]
            if len(user) < n_splits or len(user) < int(minimum/step_size):
                ids_remove_users.append(ids_[i])
                continue

        # remove users that have few samples
        df = df[['id', 'x', 'y']].query("id not in " + str(ids_remove_users))

        x_users = df['x'].tolist()
        kf = KFold(n_splits=n_splits)
        users_train_indexes = [None] * n_splits
        users_test_indexes = [None] * n_splits
        for i in range(len(x_users)):
            user = x_users[i]

            j = 0

            for train_indexes, test_indexes in kf.split(user):
                if users_train_indexes[j] is None:
                    users_train_indexes[j] = [train_indexes]
                    users_test_indexes[j] = [test_indexes]
                else:
                    users_train_indexes[j].append(train_indexes)
                    users_test_indexes[j].append(test_indexes)
                j += 1

        print("treino", len(users_train_indexes))
        print("fold 0: ", len(users_train_indexes[0][1]), len(users_test_indexes[0][1]))
        print("fold 1: ", len(users_train_indexes[1][1]), len(users_test_indexes[1][1]))
        print("fold 2: ", len(users_train_indexes[2][1]), len(users_test_indexes[2][1]))
        print("fold 3: ", len(users_train_indexes[3][1]), len(users_test_indexes[3][1]))
        print("fold 4: ", len(users_train_indexes[4][1]), len(users_test_indexes[4][1]))

        max_userid = len(df)
        print("Quantidade de usuários: ", len(df))
        # update users id
        df['id'] = np.array([i for i in range(len(df))])
        ids_list = df['id'].tolist()
        x_list = x_users
        # print("ant")
        # print(x_list[0][0])
        # x_list = [json.loads(x_list[i]) for i in range(len(x_list))]
        y_list = df['y'].tolist()
        #y_list = [json.loads(y_list[i]) for i in range(len(y_list))]
        for i in range(len(x_list)):
            sequences_list = x_list[i]


            for j in range(len(sequences_list)):
                sequence = sequences_list[j]
                for k in range(len(sequence)):

                    if sequence[k][6] != ids_list[i]:
                        print("usuário diferente")
                        exit()


                sequences_list[j] = sequence
            x_list[i] = sequences_list

        ids = df['id'].tolist()
        x = x_list
        y = y_list

        users_trajectories = df.to_numpy()
        #users_trajectories = df
        return {'ids': ids, 'x': x, 'y': y}, users_train_indexes, users_test_indexes, max_userid

    def index_worng_samples(self, indexes, x):

        x = x[indexes]

        return x

    def run_tests_one_location_output_k_fold(self,
                                             dataset_name,
                                             users_list,
                                             users_train_index,
                                             users_test_index,
                                             n_replications: int,
                                             k_folds,
                                             model_name: str,
                                             epochs,
                                             class_weight,
                                             filename,
                                             sequences_size,
                                             base_report,
                                             number_of_categories,
                                             batch,
                                             num_users,
                                             parameters,
                                             output_dir):

        print("Número de replicações", n_replications)
        folds_histories = []
        histories = []
        iteration = 0
        seeds = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
        list_wrong_samples = []
        list_y_wrong_predicted = []
        list_y_right_labels = []
        list_indexes = []
        for i in range(k_folds):
            print("Modelo: ", model_name)
            tf.random.set_seed(seeds[iteration])
            X_train, X_test, y_train, y_test = self.extract_train_test_from_indexes_k_fold_v2(users_list=users_list,
                                                                                           users_train_indexes=
                                                                                           users_train_index[i],
                                                                                           users_test_indexes=
                                                                                           users_test_index[i],
                                                                                           step_size=sequences_size,
                                                                                           number_of_categories=number_of_categories,
                                                                                           model_name=model_name,
                                                                                           seed=seeds[iteration])

            for j in range(n_replications):
                model = self._find_model(dataset_name, model_name).build(sequences_size,
                                                           location_input_dim=number_of_categories,
                                                           num_users=num_users,
                                                           time_input_dim=48,
                                                           seed=seeds[iteration])
                history, report, indexes = self._train_and_evaluate_model(model_name,
                                                                 model,
                                                                 X_train,
                                                                 y_train,
                                                                 X_test,
                                                                 y_test,
                                                                 epochs,
                                                                 batch,
                                                                 class_weight,
                                                                 parameters,
                                                                 output_dir)
                # wrong_samples = self.index_worng_samples(wrong_indexes, X_test)
                # list_wrong_samples += wrong_samples
                # list_y_wrong_predicted += y_wrong_predicted
                # list_y_right_labels += y_right_labels
                base_report = self._add_location_report(base_report, report)
                iteration+=1
                histories.append(history)
                list_indexes.append(indexes)

            # if i == 1:
            #     break
        folds_histories.append(histories)

        return folds_histories, base_report, list_wrong_samples, list_y_wrong_predicted, list_y_right_labels, list_indexes

    def extract_train_test_from_indexes_k_fold_v2(self,
                                               users_list,
                                               users_train_indexes,
                                               users_test_indexes,
                                               step_size,
                                               number_of_categories,
                                               model_name,
                                               seed,
                                               time_num_classes=48):

        X_train_concat = []
        X_test_concat = []
        y_train_concat = []
        y_test_concat = []
        #users_list = users_list[:, 1]
        ids = users_list['ids']
        x_list = users_list['x']
        y_list = users_list['y']

        x_train_spatial = []
        x_train_temporal = []
        x_train_country = []
        x_train_distance = []
        x_train_duration = []
        x_train_week_day = []
        x_train_ids = []
        x_train_pois_ids = []
        x_train_month = []
        # garg and mfa
        x_train_adjacency = []
        x_train_week_adjacency = []
        x_train_weekend_adjacency = []
        x_train_directed_adjacency = []
        x_train_distances_matrix = []
        x_train_distances_week_matrix = []
        x_train_distances_weekend_matrix = []
        x_train_temporal_matrix = []
        x_train_durations_matrix = []
        x_train_durations_week_matrix = []
        x_train_durations_weekend_matrix = []
        x_train_score = []
        x_train_poi_category_probabilities = []

        x_test_spatial = []
        x_test_temporal = []
        x_test_country = []
        x_test_distance = []
        x_test_duration = []
        x_test_week_day = []
        x_test_ids = []
        x_test_pois_ids = []
        x_test_month = []
        # garg and mfa
        x_test_adjacency = []
        x_test_week_adjacency = []
        x_test_weekend_adjacency = []
        x_test_directed_adjacency = []
        x_test_distances_matrix = []
        x_test_distances_week_matrix = []
        x_test_distances_weekend_matrix = []
        x_test_temporal_matrix = []
        x_test_durations_matrix = []
        x_test_durations_week_matrix = []
        x_test_durations_weekend_matrix = []
        x_test_poi_category_probabilities = []
        x_test_score = []

        usuario_n = 0
        for i in range(len(ids)):
            usuario_n +=1
            #print("usuario: ", usuario_n)
            user_x = np.array(x_list[i])
            user_y = np.array(y_list[i])
            # print("sgr")
            # print(user_x)
            # print("train inde")
            # print(users_train_indexes[i])
            X_train = list(user_x[users_train_indexes[i]])
            X_test = list(user_x[users_test_indexes[i]])
            y_train = list(user_y[users_train_indexes[i]])
            y_test = list(user_y[users_test_indexes[i]])
            # filter.
            maximum_train = 200
            maximum_test = 30
            if len(X_train) > maximum_train:
                X_train = X_train[:maximum_train]
                y_train = y_train[:maximum_train]
            if len(X_test) > maximum_test:
                X_test = X_test[:maximum_test]
                y_test = y_test[:maximum_test]
            if len(y_train) == 0 or len(y_test) == 0:
                continue

            # x train
            spatial_train, temporal_train, country_train, distance_train, duration_train, week_day_train, ids_train, pois_ids_train, month_train = sequence_tuples_to_spatial_temporal_and_feature8_ndarrays(X_train)
            # x_test
            spatial_test, temporal_test, country_test, distance_test, duration_test, week_day_test, ids_test, pois_ids_test, month_test = sequence_tuples_to_spatial_temporal_and_feature8_ndarrays(X_test)

            if model_name in ['garg', 'mfa', 'poi_rgnne']:
                x = [spatial_train, temporal_train, country_train, distance_train, duration_train, week_day_train, ids_train, pois_ids_train, month_train]
                adjacency_matrix_train, distances_matrix_train, temporal_matrix_train, durations_matrix_train, score_train, poi_category_probabilities_train, score_test, poi_category_probabilities_test, directed_adjacency_matrix_train, adjacency_week_matrix_train, adjacency_weekend_matrix_train, distance_week_matrix_train, distance_weekend_matrix_train, durations_week_matrix_train, durations_weekend_matrix_train = self._generate_train_graph_matrices(x, spatial_train, pois_ids_test, number_of_categories, model_name)
                #adjacency_matrix_test = self._generate_test_graph_matrices(unweighted_adjacency_matrix_train, spatial_test, number_of_categories, model_name)
                #x = [spatial_test, temporal_test, country_test, distance_test, duration_test, week_day_test, ids_test]
                #adjacency_matrix_test, distances_matrix_test, temporal_matrix_test, durations_matrix_test = self._generate_graph_matrices(x, number_of_categories, model_name)

                x_train_adjacency += adjacency_matrix_train
                x_train_week_adjacency += adjacency_week_matrix_train
                x_train_weekend_adjacency += adjacency_weekend_matrix_train
                x_train_directed_adjacency += directed_adjacency_matrix_train
                x_train_distances_matrix += distances_matrix_train
                x_train_distances_week_matrix += distance_week_matrix_train
                x_train_distances_weekend_matrix += distance_weekend_matrix_train
                x_train_temporal_matrix += temporal_matrix_train
                x_train_durations_matrix += durations_matrix_train
                x_train_durations_week_matrix += durations_week_matrix_train
                x_train_durations_weekend_matrix += durations_weekend_matrix_train
                x_train_poi_category_probabilities += poi_category_probabilities_train
                x_train_score += score_train
                x_test_adjacency += [adjacency_matrix_train[0]]*len(spatial_test)
                x_test_week_adjacency += [adjacency_week_matrix_train[0]]*len(spatial_test)
                x_test_weekend_adjacency += [adjacency_weekend_matrix_train[0]]*len(spatial_test)
                x_test_directed_adjacency += [directed_adjacency_matrix_train[0]]*len(spatial_test)
                #x_test_adjacency += adjacency_matrix_test
                x_test_distances_matrix += [distances_matrix_train[0]]*len(spatial_test)
                x_test_temporal_matrix += [temporal_matrix_train[0]]*len(spatial_test)
                x_test_durations_week_matrix += [durations_week_matrix_train[0]]*len(spatial_test)
                x_test_durations_matrix += [durations_matrix_train[0]]*len(spatial_test)
                x_test_durations_weekend_matrix += [durations_weekend_matrix_train[0]]*len(spatial_test)
                x_test_distances_week_matrix += [distance_week_matrix_train[0]]*len(spatial_test)
                x_test_distances_weekend_matrix += [durations_weekend_matrix_train[0]]*len(spatial_test)
                x_test_poi_category_probabilities += poi_category_probabilities_test
                x_test_score += score_test

                # print("tamanho treino: ", len(adjacency_matrix_train), len(distances_matrix_train))
                # print("tamanho teste: ", len(adjacency_matrix_test), len([distances_matrix_train[0]]*len(spatial_test)))



            x_train_spatial += spatial_train
            x_train_temporal += temporal_train
            #x_train_country += country_train
            x_train_distance += distance_train
            x_train_duration += duration_train
            #x_train_week_day += duration_train
            x_train_ids += ids_train
            x_train_pois_ids += pois_ids_train
            #x_train_month += month_train
            # x test
            #spatial, temporal, country, distance, duration, week_day, ids = sequence_tuples_to_spatial_temporal_and_feature6_ndarrays(X_test)
            x_test_spatial += spatial_test
            x_test_temporal += temporal_test
            #x_test_country += country_test
            x_test_distance += distance_test
            x_test_duration += duration_test
            #x_test_week_day += week_day_test
            x_test_ids += ids_test
            x_test_pois_ids += pois_ids_test
            #x_test_month += month_test

            if len(y_train) == 0:
                continue
            # X_train_concat = X_train_concat + X_train
            # X_test_concat = X_test_concat + X_test
            y_train_concat = y_train_concat + y_train
            y_test_concat = y_test_concat + y_test

        if model_name in ['garg', 'mfa', 'poi_rgnne']:
            X_train = [np.array(x_train_spatial), np.array(x_train_temporal),
                       np.array(x_train_distance), np.array(x_train_duration),
                       np.array(x_train_ids), np.array(x_train_pois_ids),
                       np.array(x_train_adjacency),
                       np.array(x_train_distances_matrix), np.array(x_train_temporal_matrix),
                       np.array(x_train_durations_matrix), np.array(x_train_score), np.array(x_train_poi_category_probabilities),
                       np.array(x_train_directed_adjacency),
                       np.array(x_train_week_adjacency), np.array(x_train_weekend_adjacency),
                       np.array(x_train_distances_week_matrix), np.array(x_train_distances_weekend_matrix),
                       np.array(x_train_durations_week_matrix), np.array(x_train_durations_weekend_matrix)]
            X_test = [np.array(x_test_spatial), np.array(x_test_temporal),
                      np.array(x_test_distance), np.array(x_test_duration),
                      np.array(x_test_ids), np.array(x_test_pois_ids),
                      np.array(x_test_adjacency),
                      np.array(x_test_distances_matrix), np.array(x_test_temporal_matrix),
                      np.array(x_test_durations_matrix), np.array(x_test_score), np.array(x_test_poi_category_probabilities),
                      np.array(x_test_directed_adjacency),
                      np.array(x_test_week_adjacency), np.array(x_test_weekend_adjacency), np.array(x_test_distances_week_matrix),
                      np.array(x_test_distances_weekend_matrix), np.array(x_test_durations_week_matrix), np.array(x_test_durations_weekend_matrix)]
        else:
            X_train = [np.array(x_train_spatial), np.array(x_train_temporal), np.array(x_train_distance), np.array(x_train_duration), np.array(x_train_week_day), np.array(x_train_ids)]
            X_test = [np.array(x_test_spatial), np.array(x_test_temporal),  np.array(x_test_distance), np.array(x_test_duration), np.array(x_test_week_day), np.array(x_test_ids)]

        y_train = y_train_concat
        y_test = y_test_concat

        # df = pd.DataFrame({'x_train': X_train, 'y_train': y_train}).sample(frac=1, random_state=2)
        #
        # df_test = pd.DataFrame({'x_test': X_test, 'y_test': y_test}).sample(frac=1, random_state=2)
        # X_train = df['x_train'].tolist()
        # X_test = df_test['x_test'].tolist()
        # y_train = df['y_train'].tolist()
        # y_test = df_test['y_test'].tolist()
        # Remove hours. Currently training without events hour
        # y_train = remove_hour_from_sequence_y(y_train)
        # y_test = remove_hour_from_sequence_y(y_test)

        X_train, y_train = self._shuffle(X_train, y_train, seed, 10)
        X_test, y_test = self._shuffle(X_test, y_test, seed, 10)

        # Sequence tuples to [spatial[,step_size], temporal[,step_size]] ndarray. Use with embedding layer.
        # X_train = sequence_tuples_to_spatial_temporal_and_feature6_ndarrays(X_train)
        # X_test = sequence_tuples_to_spatial_temporal_and_feature6_ndarrays(X_test)

        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        # Convert integers to one-hot-encoding. It is important to convert the y (labels) to that.
        #print(y_train.tolist())
        y_train = np_utils.to_categorical(y_train, num_classes=number_of_categories)
        y_test = np_utils.to_categorical(y_test, num_classes=number_of_categories)

        # when using multiple output [location, hour]
        y_train = [y_train]
        y_test = [y_test]

        return X_train, X_test, y_train, y_test

    def _add_location_report(self, location_report, report):
        for l_key in report:
            if l_key == 'accuracy':
                location_report[l_key].append(report[l_key])
                continue
            for v_key in report[l_key]:
                location_report[l_key][v_key].append(report[l_key][v_key])

        return location_report

    def _find_model(self, dataset_name, model_name):

        if dataset_name == "users_steps":
            if model_name == "serm":
                return SERMUsersSteps()
            elif model_name == "map":
                return MAPUsersSteps()
            elif model_name == "stf":
                return STFUsersSteps()
            elif model_name == "mfa":
                return MFARNNUsersSteps()
            elif model_name == "next":
                return NEXTUsersSteps()
            elif model_name == "garg":
                return GARGUsersSteps()
        elif dataset_name == "gowalla":
            if model_name == "serm":
                return SERM()
            elif model_name == "map":
                return MAP()
            elif model_name == "stf":
                return STF()
            elif model_name == "mfa":
                return MFA_RNN()
            elif model_name == "poi_rgnne":
                return POI_RGNNE()
            elif model_name == "next":
                return NEXT()
            elif model_name == "garg":
                return GARG()

    def _train_and_evaluate_model(self,
                                  model_name,
                                  model,
                                  X_train,
                                  y_train,
                                  X_test,
                                  y_test,
                                  epochs,
                                  batch,
                                  class_weight,
                                  parameters,
                                  output_dir):

        logdir = output_dir + "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
        if model_name not in ['gargs']:
            model.compile(optimizer=parameters['optimizer'], loss=parameters['loss'],
                          metrics=tf.keras.metrics.CategoricalAccuracy(name="acc"))
            #print("Quantidade de instâncias de entrada (train): ", np.array(X_train).shape)
            #print("Quantidade de instâncias de entrada (test): ", np.array(X_test).shape)


            hi = model.fit(X_train,
                           y_train,
                           validation_data=(X_test, y_test),
                           batch_size=batch,
                           epochs=epochs,
                           callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
        else:
            loss_fn = tf.keras.losses.CategoricalCrossentropy()

            # Training step
            @tf.function
            def train():
                with tf.GradientTape() as tape:
                    predictions = model(X_train, training=True)
                    loss = loss_fn(y_train, predictions)
                    loss += sum(model.losses)
                gradients = tape.gradient(loss, model.trainable_variables)
                parameters['optimizer'].apply_gradients(zip(gradients, model.trainable_variables))
                return loss

        #print("summary: ", model.summary())
        # print("history: ", h)

        y_predict_location = model.predict(X_test, batch_size=batch)

        entropies = self.entropy_of_predictions(y_predict_location)

        scores = model.evaluate(X_test, y_test, verbose=0)
        # location_acc = scores
        # print(scores)

        # To transform one_hot_encoding to list of integers, representing the locations
        print("------------- Location ------------")
        y_predict_location = one_hot_decoding(y_predict_location)
        y_test_location = one_hot_decoding(y_test[0])
        right_indexes = self.indexes_of_right_predicted_samples(y_predict_location, y_test_location)

        report = skm.classification_report(y_test_location, y_predict_location, output_dict=True)
        wrong_indexes, right_indexes = self.indexes_of_wrong_predicted_samples(y_predict_location, y_test_location)
        y_wrong_predicted = y_predict_location[wrong_indexes]
        y_wrong_labels = y_test_location[wrong_indexes]
        y_righ_predicted = y_predict_location[right_indexes]
        y_right_labels = y_test_location[right_indexes]
        entropy_right = entropies[right_indexes]
        entropy_wrong = entropies[wrong_indexes]
        print("Relatorio")
        print(report)
        print("entropia certo: ", len(entropy_right), len(entropy_right[entropy_right>0]), entropy_right[entropy_right>0].mean(),
              " entropia errado: ", len(entropy_wrong), len(entropy_wrong[entropy_wrong>0]),  entropy_wrong[entropy_wrong>0].mean())
        #return hi.history, report, wrong_indexes, y_wrong_predicted, y_right_labels
        return hi.history, report, [y_wrong_predicted, y_wrong_labels, y_righ_predicted, y_right_labels, entropy_right, entropy_wrong]

    def entropy_of_predictions(self, predictions):

        entropies = []
        for i in range(len(predictions)):
            maximum = max(predictions[i])
            minimum = min(predictions[i])
            normalized = []
            for j in range(len(predictions[i])):
                value = ((predictions[i][j]-minimum)/(maximum - minimum))
                normalized.append(value)
            out = entropy(normalized)
            # if out < 0:
            #     print("menos: ", out, " lis: ", predictions[i])
            #
            #     exit()
            entropies.append(out)

        return np.array(entropies)



    def indexes_of_wrong_predicted_samples(self, y_predicted, y_label):

        indexes = []
        right_indexes = []

        for i in range(len(y_predicted)):

            predicted = y_predicted[i]
            label = y_label[i]
            if predicted != label:
                indexes.append(i)
            else:
                right_indexes.append(i)

        return indexes, right_indexes

    def indexes_of_right_predicted_samples(self, y_predicted, y_label):

        indexes = []

        for i in range(len(y_predicted)):

            predicted = y_predicted[i]
            label = y_label[i]
            if predicted == label:
                indexes.append(i)

        return indexes

    def output_dir(self, output_base_dir, dataset_type, category_type, model_name=""):

        return output_base_dir+dataset_type+category_type+model_name

    def categories_distances_matrix_preprocessing(self, categories_distance_matrix):
        sigma = 10


        for i in range(len(categories_distance_matrix)):

            for j in range(len(categories_distance_matrix[i])):

                d_cc = categories_distance_matrix[i][j]
                if d_cc <= 0:
                    d_cc = 0
                else:
                    d_cc = d_cc * d_cc
                    d_cc = -(d_cc / (sigma * sigma))

                categories_distance_matrix[i][j] = math.exp(d_cc)

        return categories_distance_matrix

    def _generate_train_week_weekend_duration_distance_graph_matrices(self, n_categories, category, temporal, distance, duration):

        adjacency_week = [[0 for j in range(n_categories)] for i in range(n_categories)]
        adjacency_weekend = [[0 for j in range(n_categories)] for i in range(n_categories)]

        distance_week = [[[] for j in range(n_categories)] for i in range(n_categories)]
        distance_weekend = [[[] for j in range(n_categories)] for i in range(n_categories)]

        duration_week = [[[] for j in range(n_categories)] for i in range(n_categories)]
        duration_weekend = [[[] for j in range(n_categories)] for i in range(n_categories)]

        for i in range(len(category)):

            category_sequence = category[i]
            temporal_sequence = temporal[i]
            distance_sequence = distance[i]
            duration_sequence = duration[i]

            for j in range(1, len(temporal_sequence)):

                hour = temporal_sequence[j]
                from_category = int(category_sequence[j-1])
                to_category = int(category_sequence[j])
                distance_value = distance_sequence[j]
                duration_value = duration_sequence[j]
                if hour < 24:

                    adjacency_week, distance_week, duration_week = self._generate_train_week_weekend_graph_matrix(from_category, to_category, adjacency_week, distance_week, duration_week, distance_value, duration_value)

                else:

                    adjacency_weekend, distance_weekend, duration_weekend = self._generate_train_week_weekend_graph_matrix(from_category, to_category, adjacency_weekend, distance_weekend, duration_weekend, distance_value, duration_value)

        distance_week = self._summarize_categories_distance_matrix(distance_week)
        distance_weekend = self._summarize_categories_distance_matrix(distance_weekend)
        duration_week = self._summarize_categories_distance_matrix(duration_week)
        duration_weekend = self._summarize_categories_distance_matrix(duration_weekend)

        return adjacency_week, adjacency_weekend, distance_weekend, distance_weekend, duration_week, duration_weekend

    def _generate_train_week_weekend_graph_matrix(self, from_category, to_category, adjacency_matrix, distance_matrix, duration_matrix, distance_value, duration_value):

        # direct
        adjacency_matrix[from_category][to_category] += 1
        distance_matrix[from_category][to_category].append(distance_value)
        duration_matrix[from_category][to_category].append(duration_value)
        # undirect
        adjacency_matrix[to_category][from_category] += 1
        distance_matrix[to_category][from_category].append(distance_value)
        duration_matrix[to_category][from_category].append(duration_value)

        return adjacency_matrix, distance_matrix, duration_matrix



    def _generate_train_graph_matrices(self, x_train, sequence_spatial_train, pois_ids_test, n_categories, model_name):

        minimum = 0.001

        # np.asarray(spatial), np.asarray(temporal), np.array(country), np.array(distance), np.array(duration), np.array(week_day), np.asarray(ids)
        spatial, temporal, country, distance, duration, week_day, ids, pois_ids, month = x_train

        # week weekend matrices
        adjacency_week, adjacency_weekend, distance_week, distance_weekend, duration_week, duration_weekend = self._generate_train_week_weekend_duration_distance_graph_matrices(n_categories, spatial, temporal, distance, duration)

        # PoiXCategory
        unique_pois_ids = pd.Series(np.array(pois_ids).flatten()).unique().tolist()
        pois_categories_matrix = {int(unique_pois_ids[i]): [0 for j in range(n_categories)] for i in range(len(unique_pois_ids))}

        for i in range(len(pois_ids)):

            categories = spatial[i]
            pois_id = pois_ids[i]


            for j in range(len(categories)):
                category = int(categories[j])
                poi_id = int(pois_id[j])
                pois_categories_matrix[poi_id][category] += 1

        for i in range(len(unique_pois_ids)):

            poi_id = unique_pois_ids[i]
            total = sum(pois_categories_matrix[poi_id])
            pois_categories_matrix[poi_id] = list(np.array(pois_categories_matrix[poi_id])/total)

        sequences_poi_category_train = []
        for i in range(len(pois_ids)):

            pois_id = pois_ids[i]
            sequence = []
            for j in range(len(pois_id)):
                poi_id = int(pois_id[j])
                sequence.append(pois_categories_matrix[poi_id])

            sequences_poi_category_train.append(sequence)

        sequences_poi_category_test = []
        for i in range(len(pois_ids_test)):

            pois_id = pois_ids_test[i]
            sequence = []
            for j in range(len(pois_id)):
                poi_id = int(pois_id[j])
                if pois_id not in list(pois_categories_matrix.keys()):
                    pois_categories_matrix[poi_id] = [1/n_categories for j in range(n_categories)]
                sequence.append(pois_categories_matrix[poi_id])

            sequences_poi_category_test.append(sequence)

        score_train, poi_category_probabilities_train = self.poi_category_matrix(sequences_poi_category_train, spatial)
        score_test, poi_category_probabilities_test = self.poi_category_matrix(sequences_poi_category_test, spatial)




        categories_distances_matrix = [[[] for j in range(n_categories)] for i in range(n_categories)]
        categories_durations_matrix = [[[] for j in range(n_categories)] for i in range(n_categories)]
        categories_adjacency_matrix = [[0 for j in range(n_categories)] for i in range(n_categories)]
        categories_directed_adjacency_matrix = [[0 for j in range(n_categories)] for i in range(n_categories)]
        categories_temporal_matrix = [[0 for j in range(48)] for i in range(n_categories)]

        step_size = len(spatial[0])
        original_size = len(spatial)
        spatial = np.array(spatial, dtype='int').flatten()
        temporal = np.array(temporal, dtype='int').flatten()
        distance = np.array(distance).flatten()
        duration = np.array(duration).flatten()
        for i in range(1, len(spatial)):
            category = spatial[i]
            hour = temporal[i]

            pre_category = spatial[i - 1]
            #print("teeee", category, pre_category)
            categories_distances_matrix[category][pre_category].append(distance[i])
            categories_distances_matrix[pre_category][category].append(distance[i])
            categories_adjacency_matrix[category][pre_category] += 1
            categories_directed_adjacency_matrix[pre_category][category] += 1
            categories_adjacency_matrix[pre_category][category] += 1
            categories_temporal_matrix[category][hour] += 1
            categories_durations_matrix[category][pre_category].append(duration[i])
            categories_durations_matrix[pre_category][category].append(duration[i])

        categories_distances_matrix = self._summarize_categories_distance_matrix(categories_distances_matrix)
        categories_durations_matrix = self._summarize_categories_distance_matrix(categories_durations_matrix)

        categories_adjacency_matrix = np.array(categories_adjacency_matrix) + minimum
        categories_directed_adjacency_matrix = np.array(categories_directed_adjacency_matrix) + minimum
        original_categories_adjacency_matrix = categories_adjacency_matrix
        adjacency_week = np.array(adjacency_week) + minimum
        adjacency_weekend = np.array(adjacency_weekend) + minimum

        # weight adjacency matrix based on each sequence
        list_weighted_adjacency_matrices = []
        list_weighted_directed_adjacency_matrices = []
        list_weighted_adjacency_week_matrices = []
        list_weighted_adjacency_weekend_matrices = []
        for i in range(0, len(sequence_spatial_train)):
            sequence = sequence_spatial_train[i]

            # for j in range(1, len(sequence)):
            #     category = int(sequence[j])
            #     pre_category = int(sequence[j-1])
            #     #print("en", category, pre_category)
            #     weighted_adjacency_matrix = copy.copy(categories_adjacency_matrix)
            #     #weighted_adjacency_matrix[pre_category][category] -= 0
            #     weighted_adjacency_matrix[category][pre_category] -= 0
            # #weighted_adjacency_matrix[weighted_adjacency_matrix<0] = minimum
            if model_name == 'garg':
                weighted_adjacency_matrix = GCNConv.preprocess(categories_adjacency_matrix)
            elif model_name in ['mfa', 'poi_rgnne']:
                #weighted_adjacency_matrix = DiffusionConv.preprocess(weighted_adjacency_matrix)
                weighted_adjacency_matrix = GCNConv.preprocess(categories_adjacency_matrix)
                categories_directed_adjacency_matrix = DiffusionConv.preprocess(categories_directed_adjacency_matrix)
                adjacency_week = GCNConv.preprocess(adjacency_week)
                adjacency_weekend = GCNConv.preprocess(adjacency_weekend)
            list_weighted_adjacency_matrices.append(weighted_adjacency_matrix)
            list_weighted_directed_adjacency_matrices.append(categories_directed_adjacency_matrix)
            list_weighted_adjacency_week_matrices.append(adjacency_week)
            list_weighted_adjacency_weekend_matrices.append(adjacency_weekend)

        adjacency_matrix = list_weighted_adjacency_matrices
        #adjacency_matrix = [categories_adjacency_matrix]*original_size

        distances_matrix = [categories_distances_matrix]*original_size
        temporal_matrix = [categories_temporal_matrix]*original_size
        durations_matrix = [categories_durations_matrix]*original_size
        distance_week = [distance_week]*original_size
        distance_weekend = [distance_weekend]*original_size
        duration_week = [duration_week]*original_size
        duration_weekend = [duration_weekend]*original_size
        #print("compara", len(adjacency_matrix), len(distances_matrix))

        return [adjacency_matrix, distances_matrix, temporal_matrix, durations_matrix, score_train,
                poi_category_probabilities_train, score_test, poi_category_probabilities_test,
                list_weighted_directed_adjacency_matrices, list_weighted_adjacency_week_matrices,
                list_weighted_adjacency_weekend_matrices,  distance_week, distance_weekend, duration_week, duration_weekend]

    def poi_category_matrix(self, sequences_poi_category, categories):

        # now, for each sequece of poi x category matrices we will extract the score of the user and keep only the last
        # row of the matrix. This row will be multiplied by the score of the user in the category-aware layer.
        # 1 - train
        scores = []
        lasts_rows = []

        for i in range(len(sequences_poi_category)):
            # list of visited categories
            local_categories = categories[i]
            last_category = local_categories[-1]
            local_categories = local_categories[1:]

            matrix_poi_category = sequences_poi_category[i]
            last_row = matrix_poi_category[-1]
            lasts_rows.append(matrix_poi_category)
            probabilities_to_category = []
            for j in range(len(matrix_poi_category)):
                probabilities_to_category.append(np.argmax(matrix_poi_category[j]))

            probabilities_to_category = probabilities_to_category[:2]
            #print("dentro", local_categories, "\n\n fora", probabilities_to_category)
            score = skm.accuracy_score(local_categories, probabilities_to_category)
            scores.append(score)

        return scores, lasts_rows

    def _generate_test_graph_matrices(self, categories_adjacency_matrix, sequence_spatial_test, n_categories, model_name):

        list_weighted_adjacency_matrices = []
        for i in range(0, len(sequence_spatial_test)):
            sequence = sequence_spatial_test[i]

            for j in range(len(sequence)):
                category = int(sequence[j])
                pre_category = int(sequence[j - 1])
                #print("teste", category)
                weighted_adjacency_matrix = copy.copy(categories_adjacency_matrix)
                weighted_adjacency_matrix[pre_category][category] -= 0
                weighted_adjacency_matrix[category][pre_category] -= 0
            weighted_adjacency_matrix[weighted_adjacency_matrix < 0] = 0.001
            if model_name == 'garg':
                weighted_adjacency_matrix = GCNConv.preprocess(weighted_adjacency_matrix)
            elif model_name in ['mfa', 'poi_rgnne']:
                weighted_adjacency_matrix = ARMAConv.preprocess(weighted_adjacency_matrix)
            list_weighted_adjacency_matrices.append(weighted_adjacency_matrix)

        # wight adjacency matrix based on each sequence


        adjacency_matrix = list_weighted_adjacency_matrices

        return adjacency_matrix


    def _summarize_categories_distance_matrix(self, categories_distances_matrix):
        sigma = 10
        categories_distances_list = []
        for row in range(len(categories_distances_matrix)):

            category_distances_list = []
            for column in range(len(categories_distances_matrix[row])):

                values = categories_distances_matrix[row][column]

                if len(values) == 0:
                    categories_distances_matrix[row][column] = 0
                    category_distances_list.append(0)
                else:

                    d_cc = st.median(values)
                    categories_distances_matrix[row][column] = d_cc

        return categories_distances_matrix

    def _distance_importance(self, distance):

        distance = distance * distance
        distance = -(distance / (self.distance_sigma * self.distance_sigma))
        distance = math.exp(distance)

        return distance

    def _duration_importance(self, duration):

        duration = duration * duration
        duration = -(duration / (self.duration_sigma * self.duration_sigma))
        duration = math.exp(duration)

        return duration

    def _shuffle(self, x, y, seed, score_column_index):

        columns = [i for i in range(len(x))]
        data_dict = {}
        print("nnnn")
        for i in range(len(x)):
            feature = x[i].tolist()

            if i == score_column_index:
                print("feature: ", len(feature), feature[0])
                for j in range(len(feature)):
                    feature[j] = str(feature[j])
                    data_dict[columns[i]] = feature
                continue

            for j in range(len(feature)):
                feature[j] = str(list(feature[j]))

            data_dict[columns[i]] = feature

        data_dict['y'] = y


        df = pd.DataFrame(data_dict).sample(frac=1., random_state=seed)

        y = df['y'].to_numpy()
        x_new = []

        for i in range(len(columns)):
            feature = df[columns[i]].tolist()
            for j in range(len(feature)):
                feature[j] = json.loads(feature[j])
            x_new.append(np.array(feature))

        return x_new, y
