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
from spektral.layers.convolutional.gcn_conv import GCNConv
from spektral.layers.convolutional.arma_conv import ARMAConv

from extractor.file_extractor import FileExtractor
from foundation.util.next_poi_category_prediction_util import sequence_to_x_y, \
    sequence_tuples_to_spatial_temporal_and_feature6_ndarrays, \
    remove_hour_from_sequence_y
from foundation.util.nn_preprocessing import one_hot_decoding

from model.next_poi_category_prediction_models.users_steps.serm.model import SERMUsersSteps
from model.next_poi_category_prediction_models.users_steps.map.model import MAPUsersSteps
from model.next_poi_category_prediction_models.users_steps.stf.model import STFUsersSteps
from model.next_poi_category_prediction_models.users_steps.mfa_rnnuserssteps import MFA_RNNUsersSteps
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

    def read_sequences(self, filename, n_splits, model_name, number_of_categories, step_size):
        # 7000
        minimum = 60
        max_size = 4000
        df = self.file_extractor.read_csv(filename)
        df['sequence'] = df['sequence'].apply(lambda e: self._sequence_to_list(e))
        df['total'] = self._add_total(df['sequence'])
        df = df.sort_values(by='total', ascending=False)
        print(df['total'].describe())
        df = df.query("total >= " + str(minimum))
        print("usuarios com mais de " + str(minimum), len(df))
        df = df.sample(n=900, random_state=0)
        print(df)

        users_trajectories = df['sequence'].to_numpy()
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
            # first = pd.Series(categories[:int(categories_size/2)]).unique().tolist()
            # second = pd.Series(categories[int(categories_size/2):]).unique().tolist()
            # if len(first) != number_of_categories and len(second) != number_of_categories:
            #     new_sequences.append([])
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

                if distance > 50:
                    distance = 50
                if duration > 48:
                    duration = 48
                countries[country] = 0
                if country > max_country:
                    max_country = country
                if distance > max_distance:
                    max_distance = distance
                if duration > max_duration:
                    max_duration = duration
                distance = self._distance_importance(distance)
                duration = self._duration_importance(duration)
                new_sequence.append([location_category_id, hour, country, distance, duration, week_day, user_id])

            x, y = sequence_to_x_y(new_sequence, step_size)
            y = remove_hour_from_sequence_y(y)

            user_df = pd.DataFrame({'x': x, 'y': y}).sample(frac=1, random_state=1)
            x = user_df['x'].tolist()
            y = user_df['y'].tolist()

            x_list.append(x)
            y_list.append(y)

        print("quantidade usuarios: ", len(users_ids))
        print("quantidade se: ", len(x_list))
        print("maior pais: ", max_country)
        print("maior distancia: ", max_distance)
        print("maior duracao: ", max_duration)
        df['x'] = np.array(x_list)
        df['y'] = np.array(y_list)
        df = df[['id', 'x', 'y']]
        df = df.sample(frac=1, random_state=1)

        # df = self.file_extractor.read_csv(
        #     "/media/claudio/Data/backup_linux/Documentos/pycharmprojects/masters_research/location_sequence_48hours_10mil_usuarios.csv")
        # print("Colunas")
        # print(df)
        # # reindex ids
        # df['id'] = np.array([i for i in range(len(df))])
        # df['sequence'] = df['location_sequence'].apply(lambda e: self._sequence_to_list(e))
        # users_ids = df['user_id'].tolist()
        # sequences = df['sequence'].tolist()
        # new_users_sequences = []
        # for i in range(len(users_ids)):
        #
        #     user_id = users_ids[i]
        #     sequence = sequences[i]
        #     new_sequence = []
        #     for j in range(len(sequence)):
        #         hour = int(sequence[j][1])
        #         location = int(sequence[j][2])
        #         datatime_str = sequence[j][0]
        #         datetime = dt.datetime.strptime(datatime_str, '%Y-%m-%d %H:%M:%S')
        #         if datetime.weekday() < 5:
        #             day_type = 0
        #         else:
        #             day_type = 1
        #         new_sequence.append([location, hour, day_type, user_id])
        #     new_users_sequences.append(new_sequence)
        #
        # df['sequence'] = np.array(new_users_sequences)

        print("paises: ", len(list(countries.keys())))

        kf = KFold(n_splits=n_splits)
        users_train_indexes = [None] * n_splits
        users_test_indexes = [None] * n_splits

        # remove users that have few samples
        ids_remove_users = []
        ids_ = df['id'].tolist()
        x_list = df['x'].tolist()
        #x_list = [json.loads(x_list[i]) for i in range(len(x_list))]
        for i in range(df.shape[0]):
            user = x_list[i]
            j = 0
            if len(user) < n_splits or len(user) < int(minimum/step_size):
                ids_remove_users.append(ids_[i])
                continue
            for train_indexes, test_indexes in kf.split(user):
                if users_train_indexes[j] is None:
                    users_train_indexes[j] = [train_indexes]
                    users_test_indexes[j] = [test_indexes]
                else:
                    users_train_indexes[j].append(train_indexes)
                    users_test_indexes[j].append(test_indexes)
                j+=1


        print("treino", len(users_train_indexes))
        print("fold 0: ", len(users_train_indexes[0][0]), len(users_test_indexes[0][0]))
        print("fold 1: ", len(users_train_indexes[1][0]), len(users_test_indexes[1][0]))
        print("fold 2: ", len(users_train_indexes[2][0]), len(users_test_indexes[2][0]))
        print("fold 3: ", len(users_train_indexes[3][0]), len(users_test_indexes[3][0]))
        print("fold 4: ", len(users_train_indexes[4][0]), len(users_test_indexes[4][0]))



        # remove users that have few samples
        df = df[['id', 'x', 'y']].query("id not in " + str(ids_remove_users))
        max_userid = len(df)
        print("Quantidade de usuários: ", len(df))
        # update users id
        df['id'] = np.array([i for i in range(len(df))])
        ids_list = df['id'].tolist()
        x_list = df['x'].tolist()
        # print("ant")
        # print(x_list[0][0])
        # x_list = [json.loads(x_list[i]) for i in range(len(x_list))]
        y_list = df['y'].tolist()
        print("ref")
        print(y_list[0])
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

        print("final")
        print(x[0])
        print(df)

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
                history, report = self._train_and_evaluate_model(model,
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
            spatial_train, temporal_train, country_train, distance_train, duration_train, week_day_train, ids_train = sequence_tuples_to_spatial_temporal_and_feature6_ndarrays(X_train)
            # x_test
            spatial_test, temporal_test, country_test, distance_test, duration_test, week_day_test, ids_test = sequence_tuples_to_spatial_temporal_and_feature6_ndarrays(X_test)

            if model_name in ['garg', 'mfa']:
                x = [spatial_train, temporal_train, country_train, distance_train, duration_train, week_day_train, ids_train]
                adjacency_matrix_train, distances_matrix_train, temporal_matrix_train, durations_matrix_train = self._generate_graph_matrices(x, number_of_categories, model_name)
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
            # x test
            #spatial, temporal, country, distance, duration, week_day, ids = sequence_tuples_to_spatial_temporal_and_feature6_ndarrays(X_test)
            x_test_spatial += spatial_test
            x_test_temporal += temporal_test
            x_test_country += country_test
            x_test_distance += distance_test
            x_test_duration += duration_test
            x_test_week_day += week_day_test
            x_test_ids += ids_test

            if len(y_train) == 0:
                continue
            # X_train_concat = X_train_concat + X_train
            # X_test_concat = X_test_concat + X_test
            y_train_concat = y_train_concat + y_train
            y_test_concat = y_test_concat + y_test

        if model_name in ['garg', 'mfa']:
            X_train = [np.array(x_train_spatial), np.array(x_train_temporal), np.array(x_train_country), np.array(x_train_distance), np.array(x_train_duration), np.array(x_train_week_day), np.array(x_train_ids), np.array(x_train_adjacency), np.array(x_train_distances_matrix), np.array(x_train_temporal_matrix), np.array(x_train_durations_matrix)]
            X_test = [np.array(x_test_spatial), np.array(x_test_temporal), np.array(x_test_country), np.array(x_test_distance), np.array(x_test_duration), np.array(x_test_week_day), np.array(x_test_ids), np.array(x_test_adjacency), np.array(x_test_distances_matrix), np.array(x_test_temporal_matrix), np.array(x_test_durations_matrix)]
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
        x = users_list['x']
        y = users_list['y']

        x_train_spatial = []
        x_train_temporal = []
        x_train_country = []
        x_train_distance = []
        x_train_duration = []
        x_train_week_day = []
        x_train_ids = []
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
        # garg and mfa
        x_test_adjacency = []
        x_test_distances_matrix = []
        x_test_temporal_matrix = []
        x_test_durations_matrix = []
        usuario_n = 0
        for i in range(len(ids)):
            usuario_n +=1
            #print("usuario: ", usuario_n)
            user_x = np.asarray(x[i])
            user_y = np.asarray(y[i])
            print("sgr")
            print(user_x)
            X_train = list(user_x[users_train_indexes[i]])
            X_test = list(user_x[users_test_indexes[i]])
            y_train = list(user_y[users_train_indexes[i]])
            y_test = list(user_y[users_test_indexes[i]])
            if len(y_train) == 0 or len(y_test) == 0:
                continue

            # x train
            spatial_train, temporal_train, country_train, distance_train, duration_train, week_day_train, ids_train = sequence_tuples_to_spatial_temporal_and_feature6_ndarrays(X_train)
            # x_test
            spatial_test, temporal_test, country_test, distance_test, duration_test, week_day_test, ids_test = sequence_tuples_to_spatial_temporal_and_feature6_ndarrays(X_test)

            if model_name in ['garg', 'mfa']:
                x = [spatial_train, temporal_train, country_train, distance_train, duration_train, week_day_train, ids_train]
                adjacency_matrix_train, distances_matrix_train, temporal_matrix_train, durations_matrix_train = self._generate_graph_matrices(x, number_of_categories, model_name)
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
            # x test
            #spatial, temporal, country, distance, duration, week_day, ids = sequence_tuples_to_spatial_temporal_and_feature6_ndarrays(X_test)
            x_test_spatial += spatial_test
            x_test_temporal += temporal_test
            x_test_country += country_test
            x_test_distance += distance_test
            x_test_duration += duration_test
            x_test_week_day += week_day_test
            x_test_ids += ids_test

            if len(y_train) == 0:
                continue
            # X_train_concat = X_train_concat + X_train
            # X_test_concat = X_test_concat + X_test
            y_train_concat = y_train_concat + y_train
            y_test_concat = y_test_concat + y_test

        if model_name in ['garg', 'mfa']:
            X_train = [np.array(x_train_spatial), np.array(x_train_temporal), np.array(x_train_country), np.array(x_train_distance), np.array(x_train_duration), np.array(x_train_week_day), np.array(x_train_ids), np.array(x_train_adjacency), np.array(x_train_distances_matrix), np.array(x_train_temporal_matrix), np.array(x_train_durations_matrix)]
            X_test = [np.array(x_test_spatial), np.array(x_test_temporal), np.array(x_test_country), np.array(x_test_distance), np.array(x_test_duration), np.array(x_test_week_day), np.array(x_test_ids), np.array(x_test_adjacency), np.array(x_test_distances_matrix), np.array(x_test_temporal_matrix), np.array(x_test_durations_matrix)]
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
                return MFA_RNNUsersSteps()
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

    def _generate_graph_matrices(self, x_train, n_categories, model_name):

        minimum = 0.001

        # np.asarray(spatial), np.asarray(temporal), np.array(country), np.array(distance), np.array(duration), np.array(week_day), np.asarray(ids)
        spatial, temporal, country, distance, duration, week_day, ids = x_train

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
        if model_name == 'garg':
            categories_adjacency_matrix = GCNConv.preprocess(categories_adjacency_matrix)
        elif model_name == 'mfa':
            categories_adjacency_matrix = ARMAConv.preprocess(categories_adjacency_matrix)

        adjacency_matrix = [categories_adjacency_matrix]*original_size
        distances_matrix = [categories_distances_matrix]*original_size
        temporal_matrix = [categories_temporal_matrix]*original_size
        durations_matrix = [categories_durations_matrix]*original_size

        return [adjacency_matrix, distances_matrix, temporal_matrix, durations_matrix]


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
