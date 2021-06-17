from tensorflow.keras.layers import GRU, LSTM, Activation, Dense, Masking, Dropout, SimpleRNN, Input, Lambda, \
    Flatten, Reshape, Concatenate, Embedding, MultiHeadAttention, ActivityRegularization, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping

from configuration.next_poi_category_prediciton_configuration import NextPoiCategoryPredictionConfiguration
from model.next_poi_category_prediction_models.neural_network_base_model import NNBase

import numpy as np
import tensorflow as tf

from spektral.layers.convolutional import ARMAConv, GCNConv, GATConv
from spektral.layers.pooling import GlobalAvgPool, GlobalAttentionPool, GlobalAttnSumPool

l2_reg = 5e-5           # L2 regularization rate
drop_out_rate = 0
patience = 3

class MFARNNUsersSteps(NNBase):

    def __init__(self):
        super().__init__("mfa users steps")

    def build(self, step_size, location_input_dim, time_input_dim, num_users, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)
        location_category_input = Input((step_size,), dtype='float32', name='spatial')
        temporal_input = Input((step_size,), dtype='float32', name='temporal')
        country_input = Input((step_size,), dtype='float32', name='country')
        distance_input = Input((step_size,), dtype='float32', name='distance')
        duration_input = Input((step_size,), dtype='float32', name='duration')
        week_day_input = Input((step_size,), dtype='float32', name='week_day')
        user_id_input = Input((step_size,), dtype='float32', name='user')
        categories_distance_matrix = Input((location_input_dim, location_input_dim), dtype='float32', name='categories_distance_matrix')
        categories_temporal_matrix = Input((location_input_dim, 48), dtype='float32', name='categories_temporal_matrix')
        adjancency_matrix = Input((location_input_dim, location_input_dim), dtype='float32', name='adjacency_matrix')
        categories_durations_matrix = Input((location_input_dim, location_input_dim), dtype='float32', name='categories_durations_matrix')

        # adjancency_matrix = tf.cast(adjancency_matrix, dtype='float32')
        # categories_temporal_matrix = tf.cast(categories_temporal_matrix, dtype='float32')
        # categories_durations_matrix = tf.cast(categories_durations_matrix, dtype='float32')
        # categories_distance_matrix = tf.cast(categories_distance_matrix, dtype='float32')

        # The embedding layer converts integer encoded vectors to the specified
        # shape (none, input_lenght, output_dim) with random weights, which are
        # ajusted during the training turning helpful to find correlations between words.
        # Moreover, when you are working with one-hot-encoding
        # and the vocabulary is huge, you got a sparse matrix which is not computationally efficient.
        gru_units = 70
        emb_category= Embedding(input_dim=location_input_dim, output_dim=7, input_length=step_size)
        emb_time = Embedding(input_dim=time_input_dim, output_dim=3, input_length=step_size)
        emb_id = Embedding(input_dim=num_users, output_dim=3, input_length=step_size)
        emb_country = Embedding(input_dim=30, output_dim=2, input_length=step_size)
        emb_distance = Embedding(input_dim=51, output_dim=3, input_length=step_size)
        emb_duration = Embedding(input_dim=49, output_dim=3, input_length=step_size)
        emb_week_day = Embedding(input_dim=7, output_dim=3, input_length=step_size)

        spatial_embedding = emb_category(location_category_input)
        temporal_embedding = emb_time(temporal_input)
        id_embedding = emb_id(user_id_input)
        country_embbeding = emb_country(country_input)
        distance_embbeding = emb_distance(distance_input)
        duration_embbeding = emb_duration(duration_input)
        week_day_embbeding = emb_week_day(week_day_input)

        spatial_flatten = Flatten()(spatial_embedding)
        temporal_flatten = Flatten()(temporal_embedding)
        distance_flatten = Flatten()(distance_embbeding)
        duration_flatten = Flatten()(duration_embbeding)
        id_flatten = Flatten()(id_embedding)

        l_p_flatten = Concatenate()([spatial_flatten, temporal_flatten, distance_flatten, duration_flatten])
        l_p = Concatenate()([spatial_embedding, temporal_embedding, distance_embbeding, duration_embbeding])

        # l_p_flatten = Flatten()(l_p)
        # ids_flatten = Flatten()(id_flatten)

        # y_cup = tf.matmul(id_flatten, l_p_flatten)
        y_cup = Concatenate()([id_embedding, l_p])
        y_cup = Flatten()(y_cup)
        # y_cup = Dense(20)(y_cup)

        srnn = GRU(gru_units, return_sequences=True)(l_p)
        srnn = Dropout(0.5)(srnn)

        att = MultiHeadAttention(key_dim=2,
                                 num_heads=4,
                                 name='Attention')(srnn, srnn)

        # x_distances = GCNConv(22, activation='relu')([categories_distance_matrix, adjancency_matrix])
        # x_distances = Dropout(0.5)(x_distances)
        # x_distances = GCNConv(10, activation='relu')([x_distances, adjancency_matrix])
        # x_distances = Dropout(0.5)(x_distances)
        # x_distances = Flatten()(x_distances)
        #
        # x_durations = GCNConv(22, activation='relu')([categories_durations_matrix, adjancency_matrix])
        # x_durations = Dropout(0.5)(x_durations)
        # x_durations = GCNConv(10, activation='relu')([x_durations, adjancency_matrix])
        # x_durations = Dropout(0.5)(x_durations)
        # x_durations = Flatten()(x_durations)

        print("at", att.shape)
        att = Flatten()(att)
        y_sup = att
        y_sup = Dropout(0.5)(y_sup)
        y_sup = Dense(location_input_dim, activation='softmax')(y_sup)
        y_cup = Dropout(0.5)(y_cup)
        y_cup = Dense(location_input_dim, activation='softmax')(y_cup)
        print("y cup", y_cup.shape)
        print("y sup", y_sup.shape)
        y_up = y_sup

        model = Model(inputs=[location_category_input, temporal_input, country_input, distance_input, duration_input, week_day_input, user_id_input, adjancency_matrix, categories_distance_matrix, categories_temporal_matrix, categories_durations_matrix], outputs=[y_up], name="MFA-RNN")

        return model

    def mhsa(self, input):
        print("entrada mhsa: ", input.shape)
        att_layer = MultiHeadAttention(
            key_dim=1,
            num_heads=4,
            name='Multi-Head-self-attention',
        )(input, input)

        #att_layer = Dense(150, activation='elu')(att_layer)

        print("saida mhsa: ", att_layer.shape)

        return att_layer

    def graph_temporal_arma(self, x, adjacency):
        x = ARMAConv(22, iterations=1,
                     order=3,
                     share_weights=True,
                     dropout_rate=0,
                     activation='relu',
                     gcn_activation='relu',
                     kernel_regularizer=l2(l2_reg))([x, adjacency])

        x = Dropout(0.5)(x)

        x = ARMAConv(10, iterations=1,
                     order=1,
                     share_weights=True,
                     dropout_rate=drop_out_rate,
                     activation='relu',
                     gcn_activation='relu')([x, adjacency])
        x = Dropout(0.5)(x)

        return x

    def graph_distances_a(self, x, adjacency):
        x = ARMAConv(22, iterations=1,
                     order=2,
                     share_weights=True,
                     dropout_rate=0,
                     activation='relu',
                     gcn_activation='relu',
                     kernel_regularizer=l2(l2_reg))([x, adjacency])

        x = Dropout(0.5)(x)

        x = ARMAConv(10, iterations=1,
                     order=2,
                     share_weights=True,
                     dropout_rate=drop_out_rate,
                     activation='relu',
                     gcn_activation='relu')([x, adjacency])
        x = Dropout(0.5)(x)

        return x

    def graph_durations_a(self, x, adjacency):
        x = ARMAConv(22, iterations=1,
                     order=3,
                     share_weights=True,
                     dropout_rate=0,
                     activation='relu',
                     gcn_activation='relu',
                     kernel_regularizer=l2(l2_reg))([x, adjacency])

        x = Dropout(0.5)(x)

        x = ARMAConv(10, iterations=1,
                     order=2,
                     share_weights=True,
                     dropout_rate=drop_out_rate,
                     activation='relu',
                     gcn_activation='relu')([x, adjacency])
        x = Dropout(0.5)(x)

        return x

    def graph_temporal(self, x, adjacency):

        x = GCNConv(22, activation='relu')([x, adjacency])

        x = Dropout(0.5)(x)

        x = GCNConv(10, activation='relu')([x, adjacency])

        x = Dropout(0.5)(x)

        return x

    def graph_distances(self, x, adjacency):
        x = GCNConv(22, activation='relu')([x, adjacency])

        x = Dropout(0.5)(x)

        x = GCNConv(10, activation='relu')([x, adjacency])

        x = Dropout(0.5)(x)

        return x

    def graph_durations(self, x, adjacency):
        x = GCNConv(22, activation='relu')([x, adjacency])

        x = Dropout(0.5)(x)

        x = GCNConv(10, activation='relu')([x, adjacency])

        x = Dropout(0.5)(x)

        return x