import statistics as st
import datetime as dt
import math

import numpy as np
import json
import sklearn.metrics as skm
import tensorflow
from tensorflow.keras import utils as np_utils
from sklearn.model_selection import KFold

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
from model.next_poi_category_prediction_models.gowalla.serm.model import SERM
from model.next_poi_category_prediction_models.gowalla.map.model import MAP
from model.next_poi_category_prediction_models.gowalla.stf.model import STF
from model.next_poi_category_prediction_models.gowalla.mfa_rnn import MFA_RNN
from model.next_poi_category_prediction_models.gowalla.next.model import NEXT
from model.next_poi_category_prediction_models.gowalla.garg.model import GARG


class NextPoiCategoryPredictionDomain:


    def __init__(self, dataset_name):
        self.file_extractor = FileExtractor()
        self.dataset_name = dataset_name
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

    def read_sequences(self, filename, n_splits, model_name):

        df = self.file_extractor.read_csv(filename).head(4800)

        users_trajectories = df['sequence'].to_numpy()
        # reindex ids
        df['id'] = np.array([i for i in range(len(df))])
        df['sequence'] = df['sequence'].apply(lambda e: self._sequence_to_list(e))
        users_ids = df['id'].tolist()
        sequences = df['sequence'].tolist()
        new_sequences = []
        countries = {}
        max_country = 0
        max_distance = 0
        max_duration = 0
        for i in range(len(users_ids)):

            user_id = users_ids[i]
            sequence = sequences[i]
            new_sequence = []
            for j in range(len(sequence)):
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
                new_sequence.append([location_category_id, hour, country, distance, duration, week_day, user_id])

            new_sequences.append(new_sequence)

        print("quantidade usuarios: ", len(users_ids))
        print("quantidade se: ", len(new_sequences))
        print("maior pais: ", max_country)
        print("maior distancia: ", max_distance)
        print("maior duracao: ", max_duration)
        df['sequence'] = np.array(new_sequences)

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
        for i in range(df.shape[0]):
            user = df.iloc[i]['sequence']
            j = 0
            if len(user) < n_splits:
                ids_remove_users.append(df.iloc[i]['id'])
                continue
            for train_indexes, test_indexes in kf.split(user):
                if users_train_indexes[j] is None:
                    users_train_indexes[j] = [train_indexes]
                    users_test_indexes[j] = [test_indexes]
                else:
                    users_train_indexes[j].append(train_indexes)
                    users_test_indexes[j].append(test_indexes)
                j+=1


        print("treino", len(users_train_indexes), len(users_train_indexes[0]), len(users_train_indexes[1]))

        max_userid = df['id'].max()
        # remove users that have few samples
        df = df[['id', 'sequence']].query("id not in " + str(ids_remove_users))
        users_trajectories = df.to_numpy()
        #users_trajectories = df
        return users_trajectories, users_train_indexes, users_test_indexes, max_userid

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
                                             optimizer,
                                             output_dir):

        print("Número de replicações", n_replications)
        folds_histories = []
        histories = []
        iteration = 0
        for i in range(k_folds):
            print("Modelo: ", model_name)
            X_train, X_test, y_train, y_test = self.extract_train_test_from_indexes_k_fold(users_list=users_list,
                                                                                           users_train_indexes=
                                                                                           users_train_index[i],
                                                                                           users_test_indexes=
                                                                                           users_test_index[i],
                                                                                           step_size=sequences_size,
                                                                                           number_of_categories=number_of_categories,
                                                                                           model_name=model_name)

            for j in range(n_replications):
                model = self._find_model(dataset_name, model_name).build(sequences_size,
                                                           location_input_dim=number_of_categories,
                                                           num_users=num_users,
                                                           time_input_dim=48,
                                                           seed=iteration)
                history, report = self._train_and_evaluate_model(model,
                                                                 X_train,
                                                                 y_train,
                                                                 X_test,
                                                                 y_test,
                                                                 epochs,
                                                                 batch,
                                                                 class_weight,
                                                                 optimizer,
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
        usuario_n = 0
        for i in range(len(users_list)):
            usuario_n +=1
            print("usuario: ", usuario_n)
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
                adjacency_matrix_train, distances_matrix_train, temporal_matrix_train = self._generate_graph_matrices(x, number_of_categories)

                x = [spatial_test, temporal_test, country_test, distance_test, duration_test, week_day_test, ids_test]
                adjacency_matrix_test, distances_matrix_test, temporal_matrix_test = self._generate_graph_matrices(x, number_of_categories)

                x_train_adjacency += adjacency_matrix_train
                x_train_distances_matrix += distances_matrix_train
                x_train_temporal_matrix += temporal_matrix_train
                x_test_adjacency += adjacency_matrix_test
                x_test_distances_matrix += distances_matrix_test
                x_test_temporal_matrix += temporal_matrix_test

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
            X_train = [np.array(x_train_spatial), np.array(x_train_temporal), np.array(x_train_country), np.array(x_train_distance), np.array(x_train_duration), np.array(x_train_week_day), np.array(x_train_ids), np.array(x_train_adjacency), np.array(x_train_distances_matrix), np.array(x_train_temporal_matrix)]
            X_test = [np.array(x_test_spatial), np.array(x_test_temporal), np.array(x_test_country), np.array(x_test_distance), np.array(x_test_duration), np.array(x_test_week_day), np.array(x_test_ids), np.array(x_test_adjacency), np.array(x_test_distances_matrix), np.array(x_test_temporal_matrix)]
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
                                  optimizer,
                                  output_dir):

        logdir = output_dir + "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=logdir)

        model.compile(optimizer='adam', loss=["categorical_crossentropy"],
                      metrics=tensorflow.keras.metrics.CategoricalAccuracy(name="acc"))
        #print("Quantidade de instâncias de entrada (train): ", np.array(X_train).shape)
        #print("Quantidade de instâncias de entrada (test): ", np.array(X_test).shape)
        hi = model.fit(X_train,
                       y_train,
                       validation_data=(X_test, y_test),
                       batch_size=batch,
                       epochs=epochs)

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

    def _generate_graph_matrices(self, x_train, n_categories):

        # np.asarray(spatial), np.asarray(temporal), np.array(country), np.array(distance), np.array(duration), np.array(week_day), np.asarray(ids)
        spatial, temporal, country, distance, duration, week_day, ids = x_train

        categories_distances_matrix = [[[] for j in range(n_categories)] for i in range(n_categories)]
        categories_adjacency_matrix = [[0 for j in range(n_categories)] for i in range(n_categories)]
        categories_temporal_matrix = [[0 for j in range(48)] for i in range(n_categories)]
        step_size = len(spatial[0])
        original_size = len(spatial)
        spatial = np.array(spatial).flatten()
        temporal = np.array(temporal).flatten()
        distance = np.array(distance).flatten()
        for i in range(1, len(spatial)):
            category = spatial[i]
            hour = temporal[i]

            pre_category = spatial[i - 1]
            categories_distances_matrix[category][pre_category].append(distance[i])
            categories_distances_matrix[pre_category][category].append(distance[i])
            categories_adjacency_matrix[category][pre_category] += 1
            categories_adjacency_matrix[pre_category][category] += 1
            categories_temporal_matrix[category][hour] += 1

        categories_distances_matrix = self._summarize_categories_distance_matrix(categories_distances_matrix)

        adjacency_matrix = [categories_adjacency_matrix]*original_size
        distances_matrix = [categories_distances_matrix]*original_size
        temporal_matrix = [categories_temporal_matrix]*original_size

        return [adjacency_matrix, distances_matrix, temporal_matrix]


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
                    d_cc = d_cc * d_cc
                    d_cc = -(d_cc / (sigma * sigma))
                    categories_distances_matrix[row][column] = math.exp(d_cc)

        return categories_distances_matrix