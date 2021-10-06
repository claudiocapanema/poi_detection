import copy
import statistics as st
import datetime as dt
import math

import numpy as np
import json

import pandas as pd
import sklearn.metrics as skm
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import utils as np_utils
from sklearn.model_selection import KFold
from spektral.transforms.layer_preprocess import LayerPreprocess
from spektral.layers.convolutional import GCNConv, ARMAConv, DiffusionConv

from extractor.file_extractor import FileExtractor
from foundation.util.next_poi_category_prediction_util import sequence_to_x_y, \
    sequence_tuples_to_spatial_temporal_and_feature7_ndarrays, \
    remove_hour_from_sequence_y, sequence_to_x_y_v1
from foundation.util.nn_preprocessing import one_hot_decoding

from model.next_poi_category_prediction_models.users_steps.serm.model import SERMUsersSteps
from model.next_poi_category_prediction_models.users_steps.map.model import MAPUsersSteps
from model.next_poi_category_prediction_models.users_steps.stf.model import STFUsersSteps
from model.next_poi_category_prediction_models.users_steps.mfarnnuserssteps import MFARNNUsersSteps
from model.next_poi_category_prediction_models.users_steps.next.model import NEXTUsersSteps
from model.next_poi_category_prediction_models.gowalla.serm.serm import SERM
from model.next_poi_category_prediction_models.gowalla.map.map import MAP
from model.next_poi_category_prediction_models.gowalla.stf.stf import STF
from model.next_poi_category_prediction_models.gowalla.mfa_rnn import MFA_RNN
from model.next_poi_category_prediction_models.gowalla.next.next import NEXT
from model.next_poi_category_prediction_models.gowalla.garg.garg import GARG
from model.next_poi_category_prediction_models.users_steps.garg.garg import GARGUsersSteps


class NextPoiCategoryPredictionDomain:


    def __init__(self, dataset_name, distance_sigma, duration_sigma):
        self.file_extractor = FileExtractor()
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
            n = 1000
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
                new_sequence.append([location_category_id, hour, country, distance, duration, week_day, user_id, poi_id])

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

                    sequence[k][-1] = ids_list[i]


                sequences_list[j] = sequence
            x_list[i] = sequences_list

        ids = df['id'].tolist()
        x = x_list
        y = y_list

        users_trajectories = df.to_numpy()
        #users_trajectories = df
        return {'ids': ids, 'x': x, 'y': y}, users_train_indexes, users_test_indexes, max_userid

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
                history, report = self._train_and_evaluate_model(model_name,
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
                base_report = self._add_location_report(base_report, report)
                iteration+=1
                histories.append(history)
        folds_histories.append(histories)

        return folds_histories, base_report

    def extract_train_test_from_indexes_k_fold(self,
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
        users_list = users_list[:, 1]

        x_train_spatial = []
        x_train_temporal = []
        x_train_country = []
        x_train_distance = []
        x_train_duration = []
        x_train_week_day = []
        x_train_ids = []
        x_train_pois_ids = []
        # garg and mfa
        x_train_adjacency = []
        x_train_distances_matrix = []
        x_train_temporal_matrix = []
        x_train_durations_matrix = []

        x_test_spatial = []
        x_test_temporal = []
        x_test_country = []
        x_test_distance = []
        x_test_duration = []
        x_test_week_day = []
        x_test_ids = []
        x_test_pois_ids = []
        # garg and mfa
        x_test_adjacency = []
        x_test_distances_matrix = []
        x_test_temporal_matrix = []
        x_test_durations_matrix = []
        usuario_n = 0
        for i in range(len(users_list)):
            usuario_n +=1
            #print("usuario: ", usuario_n)
            user = np.asarray(users_list[i])
            train = user[users_train_indexes[i]]
            test = user[users_test_indexes[i]]
            X_train, y_train = sequence_to_x_y(train, step_size)
            X_test, y_test = sequence_to_x_y(test, step_size)
            if len(y_train) == 0 or len(y_test) == 0:
                continue

            # x train
            spatial_train, temporal_train, country_train, distance_train, duration_train, week_day_train, ids_train, pois_ids_train = sequence_tuples_to_spatial_temporal_and_feature7_ndarrays(X_train)
            # x_test
            spatial_test, temporal_test, country_test, distance_test, duration_test, week_day_test, ids_test, pois_ids_test = sequence_tuples_to_spatial_temporal_and_feature7_ndarrays(X_test)

            if model_name in ['garg', 'mfa']:
                print("aquii", pois_ids_train)
                x = [spatial_train, temporal_train, country_train, distance_train, duration_train, week_day_train, ids_train, pois_ids_train]
                adjacency_matrix_train, distances_matrix_train, temporal_matrix_train, durations_matrix_train = self._generate_train_graph_matrices(x, spatial_train, number_of_categories, model_name)
                #x = [spatial_test, temporal_test, country_test, distance_test, duration_test, week_day_test, ids_test]
                #adjacency_matrix_test, distances_matrix_test, temporal_matrix_test, durations_matrix_test = self._generate_graph_matrices(x, number_of_categories, model_name)

                x_train_adjacency += adjacency_matrix_train
                x_train_distances_matrix += distances_matrix_train
                x_train_temporal_matrix += temporal_matrix_train
                x_train_durations_matrix += durations_matrix_train
                x_test_adjacency += [adjacency_matrix_train[0]]*len(spatial_test)
                x_test_distances_matrix += [distances_matrix_train[0]]*len(spatial_test)
                x_test_temporal_matrix += [temporal_matrix_train[0]]*len(spatial_test)
                x_test_durations_matrix += [durations_matrix_train[0]]*len(spatial_test)

            x_train_spatial += spatial_train
            x_train_temporal += temporal_train
            x_train_country += country_train
            x_train_distance += distance_train
            x_train_duration += duration_train
            x_train_week_day += duration_train
            x_train_ids += ids_train
            x_train_pois_ids += pois_ids_train
            # x test
            #spatial, temporal, country, distance, duration, week_day, ids = sequence_tuples_to_spatial_temporal_and_feature6_ndarrays(X_test)
            x_test_spatial += spatial_test
            x_test_temporal += temporal_test
            x_test_country += country_test
            x_test_distance += distance_test
            x_test_duration += duration_test
            x_test_week_day += week_day_test
            x_test_ids += ids_test
            x_test_pois_ids += pois_ids_test

            if len(y_train) == 0:
                continue
            # X_train_concat = X_train_concat + X_train
            # X_test_concat = X_test_concat + X_test
            y_train_concat = y_train_concat + y_train
            y_test_concat = y_test_concat + y_test

        if model_name in ['garg', 'mfa']:
            X_train = [np.array(x_train_spatial), np.array(x_train_temporal), np.array(x_train_country), np.array(x_train_distance), np.array(x_train_duration), np.array(x_train_week_day), np.array(x_train_ids), np.array(x_train_pois_ids), np.array(x_train_adjacency), np.array(x_train_distances_matrix), np.array(x_train_temporal_matrix), np.array(x_train_durations_matrix)]
            X_test = [np.array(x_test_spatial), np.array(x_test_temporal), np.array(x_test_country), np.array(x_test_distance), np.array(x_test_duration), np.array(x_test_week_day), np.array(x_test_ids), np.array(x_test_pois_ids), np.array(x_test_adjacency), np.array(x_test_distances_matrix), np.array(x_test_temporal_matrix), np.array(x_test_durations_matrix)]
        else:
            X_train = [np.array(x_train_spatial), np.array(x_train_temporal), np.array(x_train_country), np.array(x_train_distance), np.array(x_train_duration), np.array(x_train_week_day), np.array(x_train_ids)]
            X_test = [np.array(x_test_spatial), np.array(x_test_temporal), np.array(x_test_country), np.array(x_test_distance), np.array(x_test_duration), np.array(x_test_week_day), np.array(x_test_ids)]

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
        y_train = remove_hour_from_sequence_y(y_train)
        y_test = remove_hour_from_sequence_y(y_test)

        X_train, y_train = self._shuffle(X_train, y_train, seed)
        X_test, y_test = self._shuffle(X_test, y_test, seed)

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
        # garg and mfa
        x_train_adjacency = []
        x_train_distances_matrix = []
        x_train_temporal_matrix = []
        x_train_durations_matrix = []
        x_train_sequences_poi_category = []

        x_test_spatial = []
        x_test_temporal = []
        x_test_country = []
        x_test_distance = []
        x_test_duration = []
        x_test_week_day = []
        x_test_ids = []
        x_test_pois_ids = []
        # garg and mfa
        x_test_adjacency = []
        x_test_distances_matrix = []
        x_test_temporal_matrix = []
        x_test_durations_matrix = []
        x_test_sequences_poi_category = []
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
            if len(y_train) == 0 or len(y_test) == 0:
                continue

            # x train
            spatial_train, temporal_train, country_train, distance_train, duration_train, week_day_train, ids_train, pois_ids_train = sequence_tuples_to_spatial_temporal_and_feature7_ndarrays(X_train)
            # x_test
            spatial_test, temporal_test, country_test, distance_test, duration_test, week_day_test, ids_test, pois_ids_test = sequence_tuples_to_spatial_temporal_and_feature7_ndarrays(X_test)

            if model_name in ['garg', 'mfa']:
                x = [spatial_train, temporal_train, country_train, distance_train, duration_train, week_day_train, ids_train, pois_ids_train]
                adjacency_matrix_train, distances_matrix_train, temporal_matrix_train, durations_matrix_train, sequences_poi_category_train, sequences_poi_category_test = self._generate_train_graph_matrices(x, spatial_train, pois_ids_test, number_of_categories, model_name)
                #adjacency_matrix_test = self._generate_test_graph_matrices(unweighted_adjacency_matrix_train, spatial_test, number_of_categories, model_name)
                #x = [spatial_test, temporal_test, country_test, distance_test, duration_test, week_day_test, ids_test]
                #adjacency_matrix_test, distances_matrix_test, temporal_matrix_test, durations_matrix_test = self._generate_graph_matrices(x, number_of_categories, model_name)

                x_train_adjacency += adjacency_matrix_train
                x_train_distances_matrix += distances_matrix_train
                x_train_temporal_matrix += temporal_matrix_train
                x_train_durations_matrix += durations_matrix_train
                x_train_sequences_poi_category += sequences_poi_category_train
                x_test_adjacency += [adjacency_matrix_train[0]]*len(spatial_test)
                #x_test_adjacency += adjacency_matrix_test
                x_test_distances_matrix += [distances_matrix_train[0]]*len(spatial_test)
                x_test_temporal_matrix += [temporal_matrix_train[0]]*len(spatial_test)
                x_test_durations_matrix += [durations_matrix_train[0]]*len(spatial_test)
                x_test_sequences_poi_category += sequences_poi_category_test

                # print("tamanho treino: ", len(adjacency_matrix_train), len(distances_matrix_train))
                # print("tamanho teste: ", len(adjacency_matrix_test), len([distances_matrix_train[0]]*len(spatial_test)))



            x_train_spatial += spatial_train
            x_train_temporal += temporal_train
            x_train_country += country_train
            x_train_distance += distance_train
            x_train_duration += duration_train
            x_train_week_day += duration_train
            x_train_ids += ids_train
            x_train_pois_ids += pois_ids_train
            # x test
            #spatial, temporal, country, distance, duration, week_day, ids = sequence_tuples_to_spatial_temporal_and_feature6_ndarrays(X_test)
            x_test_spatial += spatial_test
            x_test_temporal += temporal_test
            x_test_country += country_test
            x_test_distance += distance_test
            x_test_duration += duration_test
            x_test_week_day += week_day_test
            x_test_ids += ids_test
            x_test_pois_ids += pois_ids_test

            if len(y_train) == 0:
                continue
            # X_train_concat = X_train_concat + X_train
            # X_test_concat = X_test_concat + X_test
            y_train_concat = y_train_concat + y_train
            y_test_concat = y_test_concat + y_test

        if model_name in ['garg', 'mfa']:
            X_train = [np.array(x_train_spatial), np.array(x_train_temporal), np.array(x_train_country), np.array(x_train_distance), np.array(x_train_duration), np.array(x_train_week_day), np.array(x_train_ids), np.array(x_train_pois_ids), np.array(x_train_adjacency), np.array(x_train_distances_matrix), np.array(x_train_temporal_matrix), np.array(x_train_durations_matrix), np.array(x_train_sequences_poi_category)]
            X_test = [np.array(x_test_spatial), np.array(x_test_temporal), np.array(x_test_country), np.array(x_test_distance), np.array(x_test_duration), np.array(x_test_week_day), np.array(x_test_ids), np.array(x_test_pois_ids), np.array(x_test_adjacency), np.array(x_test_distances_matrix), np.array(x_test_temporal_matrix), np.array(x_test_durations_matrix), np.array(x_test_sequences_poi_category)]
        else:
            X_train = [np.array(x_train_spatial), np.array(x_train_temporal), np.array(x_train_country), np.array(x_train_distance), np.array(x_train_duration), np.array(x_train_week_day), np.array(x_train_ids)]
            X_test = [np.array(x_test_spatial), np.array(x_test_temporal), np.array(x_test_country), np.array(x_test_distance), np.array(x_test_duration), np.array(x_test_week_day), np.array(x_test_ids)]

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

        X_train, y_train = self._shuffle(X_train, y_train, seed)
        X_test, y_test = self._shuffle(X_test, y_test, seed)

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
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

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
                           callbacks=EarlyStopping(patience=3, restore_best_weights=True))
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

        scores = model.evaluate(X_test, y_test, verbose=0)
        # location_acc = scores
        # print(scores)

        # To transform one_hot_encoding to list of integers, representing the locations
        print("------------- Location ------------")
        y_predict_location = one_hot_decoding(y_predict_location)
        y_test_location = one_hot_decoding(y_test[0])

        report = skm.classification_report(y_test_location, y_predict_location, output_dict=True)
        print("Relatorio")
        print(report)
        return hi.history, report

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

    def _generate_train_graph_matrices(self, x_train, sequence_spatial_train, pois_ids_test, n_categories, model_name):

        minimum = 0.001

        # np.asarray(spatial), np.asarray(temporal), np.array(country), np.array(distance), np.array(duration), np.array(week_day), np.asarray(ids)
        spatial, temporal, country, distance, duration, week_day, ids, pois_ids = x_train

        # PoiXCategory
        unique_pois_ids = pd.Series(np.array(pois_ids).flatten()).unique().tolist()
        pois_categories_matrix = {unique_pois_ids[i]: [0 for j in range(n_categories)] for i in range(len(unique_pois_ids))}

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
                sequence.append(pois_categories_matrix[poi_id])

            sequences_poi_category_test.append(sequence)

        categories_distances_matrix = [[[] for j in range(n_categories)] for i in range(n_categories)]
        categories_durations_matrix = [[[] for j in range(n_categories)] for i in range(n_categories)]
        categories_adjacency_matrix = [[0 for j in range(n_categories)] for i in range(n_categories)]
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
            categories_adjacency_matrix[pre_category][category] += 1
            categories_temporal_matrix[category][hour] += 1
            categories_durations_matrix[category][pre_category].append(duration[i])
            categories_durations_matrix[pre_category][category].append(duration[i])

        categories_distances_matrix = self._summarize_categories_distance_matrix(categories_distances_matrix)
        categories_durations_matrix = self._summarize_categories_distance_matrix(categories_durations_matrix)

        categories_adjacency_matrix = np.array(categories_adjacency_matrix) + minimum
        original_categories_adjacency_matrix = categories_adjacency_matrix

        # weight adjacency matrix based on each sequence
        list_weighted_adjacency_matrices = []
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
            elif model_name == 'mfa':
                #weighted_adjacency_matrix = DiffusionConv.preprocess(weighted_adjacency_matrix)
                weighted_adjacency_matrix = GCNConv.preprocess(categories_adjacency_matrix)
            list_weighted_adjacency_matrices.append(weighted_adjacency_matrix)

        adjacency_matrix = list_weighted_adjacency_matrices
        #adjacency_matrix = [categories_adjacency_matrix]*original_size

        distances_matrix = [categories_distances_matrix]*original_size
        temporal_matrix = [categories_temporal_matrix]*original_size
        durations_matrix = [categories_durations_matrix]*original_size
        #print("compara", len(adjacency_matrix), len(distances_matrix))

        return [adjacency_matrix, distances_matrix, temporal_matrix, durations_matrix, sequences_poi_category_train, sequences_poi_category_test]

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
            elif model_name == 'mfa':
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

    def _shuffle(self, x, y, seed):

        columns = [i for i in range(len(x))]
        data_dict = {}
        for i in range(len(x)):
            feature = x[i].tolist()
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
