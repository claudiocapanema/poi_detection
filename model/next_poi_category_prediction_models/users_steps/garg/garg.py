from tensorflow.keras.layers import GRU, LSTM, Activation, Dense, Masking, Dropout, SimpleRNN, Input, Lambda, \
    Flatten, Reshape, MultiHeadAttention
from tensorflow.keras.layers import add, Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
from spektral.layers.convolutional.gcn_conv import GCNConv
import numpy as np

import tensorflow as tf


class GARGUsersSteps:

    def __init__(self):
        self.name = "garg"

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
        pois_ids_input = Input((step_size,), dtype='float32', name='pois_ids')
        categories_distance_matrix = Input((location_input_dim, location_input_dim), dtype='float32',
                                           name='categories_distance_matrix')
        categories_temporal_matrix = Input((location_input_dim, 48), dtype='float32', name='categories_temporal_matrix')
        adjancency_matrix = Input((location_input_dim, location_input_dim), dtype='float32', name='adjacency_matrix')
        categories_durations_matrix = Input((location_input_dim, location_input_dim), dtype='float32',
                                            name='categories_durations_matrix')
        sequence_poi_category_matrix = Input((step_size, location_input_dim), dtype='float32',
                                             name='sequence_poi_category_matrix')

        # The embedding layer converts integer encoded vectors to the specified
        # shape (none, input_lenght, output_dim) with random weights, which are
        # ajusted during the training turning helpful to find correlations between words.
        # Moreover, when you are working with one-hot-encoding
        # and the vocabulary is huge, you got a sparse matrix which is not computationally efficient.
        units = 60
        emb_category = Embedding(input_dim=location_input_dim, output_dim=7, input_length=step_size)
        emb_time = Embedding(input_dim=time_input_dim, output_dim=3, input_length=step_size)
        emb_id = Embedding(input_dim=num_users, output_dim=2, input_length=step_size)
        emb_country = Embedding(input_dim=30, output_dim=3, input_length=step_size)
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

        id_flatten = Dense(location_input_dim)(id_flatten)
        l_p_flatten = Dense(location_input_dim)(l_p_flatten)

        y_cup = tf.math.multiply(id_flatten, l_p_flatten)
        srnn = GRU(units, return_sequences=True)(l_p)
        srnn = Dropout(0.5)(srnn)

        att = MultiHeadAttention(key_dim=2,
                                 num_heads=1,
                                 name='Attention')(srnn, srnn)

        x = GCNConv(22, activation='relu')([categories_distance_matrix, adjancency_matrix])
        x = Dropout(0.5)(x)
        x = GCNConv(10, activation='relu')([x, adjancency_matrix])
        x = Dropout(0.5)(x)
        x = Flatten()(x)

        srnn = Flatten()(srnn)
        y_r = Dense(location_input_dim, activation='softmax')(srnn)

        att = Flatten()(att)
        att = Dense(location_input_dim)(att)
        x = Dense(location_input_dim)(x)
        y_sup = tf.math.multiply(att, x)
        y_sup = Dropout(0.3)(y_sup)
        y_sup = Dense(location_input_dim, activation='softmax')(y_sup)
        y_cup = Dropout(0.5)(y_cup)
        y_cup = Dense(location_input_dim, activation='softmax')(y_cup)
        y_up = y_cup + tf.Variable(initial_value=0.) * y_sup + y_r * tf.Variable(initial_value=1.)

        model = Model(inputs=[location_category_input, temporal_input, country_input, distance_input, duration_input, week_day_input, user_id_input, pois_ids_input, adjancency_matrix, categories_distance_matrix, categories_temporal_matrix, categories_durations_matrix, sequence_poi_category_matrix], outputs=[y_up], name="GARG_baseline")

        return model

