import ast
import datetime as dt
import pandas as pd
import numpy as np
import json
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import copy
import sklearn.metrics as skm
from keras import utils as np_utils
import tensorflow as tf
from sklearn.model_selection import KFold
import keras

from extractor.file_extractor import FileExtractor
from foundation.util.next_poi_category_prediction_util import sequence_to_x_y, \
    sequence_tuples_to_spatial_temporal_and_feature3_ndarrays, return_hour_from_sequence_y, \
    remove_hour_from_sequence_y, remove_hour_from_sequence_x
from foundation.util.nn_preprocessing import one_hot_decoding, \
    one_hot_decoding_predicted, top_k_rows, weighted_categorical_crossentropy, \
    filter_data_by_valid_category

from model.next_poi_category_prediction_models.serm.model import SERM
from model.next_poi_category_prediction_models.map.model import MAP
from model.next_poi_category_prediction_models.stf.model import STF
from model.next_poi_category_prediction_models.mfa_rnn import MFA_RNN


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
        series = json.loads(series.replace("\n", "").replace(" ", ",").replace(".","").replace(",,",",").replace("[,", "[").replace(",,",",").replace("[,", "[").replace(",,", ","))

        return series

    def read_sequences(self, filename, n_splits):

        df = self.file_extractor.read_csv(filename)

        users_trajectories = df['sequence'].to_numpy()
        # reindex ids
        df['id'] = np.array([i for i in range(len(df))])
        df['sequence'] = df['sequence'].apply(lambda e: self._sequence_to_list(e))
        users_ids = df['id'].tolist()
        sequences = df['sequence'].tolist()
        new_sequences = []
        for i in range(len(users_ids)):

            user_id = users_ids[i]
            sequence = sequences[i]

            for j in range(len(sequence)):
                sequence[j][3] = user_id

            new_sequences.append(sequence)

        print("quantidade usuarios: ", len(users_ids))
        print("quantidade se: ", len(new_sequences))
        df['sequence'] = np.array(new_sequences)

        # print("tamanho", users_trajectories.shape)
        # users_sequences = np.zeros(shape=users_trajectories.shape[0], dtype=object)
        # for i in range(users_trajectories.shape[0]):
        #     user_trajetory = self._sequence_to_list(users_trajectories[i])
        #     user_sequences = []
        #     sequence = []
        #     for j in range(len(user_trajetory)):
        #         sequence.append(user_trajetory[j])
        #         if len(sequence) == sequences_size:
        #
        #             user_sequences.append(sequence)
        #             sequence = []
        #     users_sequences[i] = np.array(user_sequences)
        #     #users_sequences[i] = np.array([df['userid'].iloc[i], user_sequences])
        #     #print(np.array([df['userid'].iloc[i], user_sequences]))
        #
        #
        #
        # users_sequences = np.array([users_ids, users_sequences]).T

        # return users_sequences

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
        print("akii", users_trajectories.shape)
        return users_trajectories, users_train_indexes, users_test_indexes, max_userid

    def run_tests_one_location_output_k_fold(self,
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

        print("testes", n_replications)
        folds_histories = []
        histories = []
        iteration = 0
        for i in range(k_folds):
            # if i > 1:
            #     continue
            X_train, X_test, y_train, y_test = self.extract_train_test_from_indexes_k_fold(users_list=users_list,
                                                                                           users_train_indexes=
                                                                                           users_train_index[i],
                                                                                           users_test_indexes=
                                                                                           users_test_index[i],
                                                                                           step_size=sequences_size,
                                                                                           number_of_categories=number_of_categories)

            for j in range(n_replications):
                model = self._find_model(model_name).build(sequences_size,
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
                                               time_num_classes=48):

        X_train_concat = []
        X_test_concat = []
        y_train_concat = []
        y_test_concat = []
        users_list = users_list[:, 1]
        for i in range(len(users_list)):
            user = np.asarray(users_list[i])
            train = user[users_train_indexes[i]]
            test = user[users_test_indexes[i]]
            X_train, y_train = sequence_to_x_y(train, step_size)
            X_test, y_test = sequence_to_x_y(test, step_size)
            if len(y_train) == 0:
                continue
            X_train_concat = X_train_concat + X_train
            X_test_concat = X_test_concat + X_test
            y_train_concat = y_train_concat + y_train
            y_test_concat = y_test_concat + y_test

        X_train = X_train_concat
        X_test = X_test_concat
        y_train = y_train_concat
        y_test = y_test_concat

        y_train_hours = return_hour_from_sequence_y(y_train)
        y_test_hours = return_hour_from_sequence_y(y_test)

        # Remove hours. Currently training without events hour
        # X_train = remove_hour_from_sequence_x(X_train)
        # X_test = remove_hour_from_sequence_x(X_test)
        y_train = remove_hour_from_sequence_y(y_train)
        y_test = remove_hour_from_sequence_y(y_test)

        # Sequence tuples to ndarray. Use it if the remove hour function was not called
        # X_train = sequence_tuples_to_ndarray_x(X_train)
        # X_test = sequence_tuples_to_ndarray_x(X_test)
        # y_train = sequence_tuples_to_ndarray_y(y_train)
        # y_test = sequence_tuples_to_ndarray_y(y_test)

        # Sequence tuples to [spatial[,step_size], temporal[,step_size]] ndarray. Use with embedding layer.
        X_train = sequence_tuples_to_spatial_temporal_and_feature3_ndarrays(X_train)
        X_test = sequence_tuples_to_spatial_temporal_and_feature3_ndarrays(X_test)

        # X_train = np.asarray(X_train)
        # X_train = X_train.reshape(X_train.shape + (1,)) # perform it if the step doesn't have hour
        y_train = np.asarray(y_train)
        # X_test = np.asarray(X_test)
        # X_test = X_test.reshape(X_test.shape + (1,))
        y_test = np.asarray(y_test)

        # Convert integers to one-hot-encoding. It is important to convert the y (labels) to that.
        print("aqu")
        print(y_train)
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

    def _find_model(self, model_name):

        if model_name == "serm":
            return SERM()
        elif model_name == "map":
            return MAP()
        elif model_name == "stf":
            return STF()
        elif model_name == "mfa":
            return MFA_RNN()

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
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        model.compile(optimizer='adam', loss=["categorical_crossentropy"],
                      weighted_metrics=[keras.metrics.Accuracy(name="acc")])

        #print("entrada", X_train)
        hi = model.fit(X_train,
                       y_train,
                       validation_data=(X_test, y_test),
                       batch_size=batch,
                       epochs=epochs)

        print("summary: ", model.summary())
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

        return hi.history, report

    def output_dir(self, output_base_dir, dataset_type, category_type, model_name=""):

        return output_base_dir+dataset_type+category_type+model_name