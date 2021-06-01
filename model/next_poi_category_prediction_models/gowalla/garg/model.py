from tensorflow.keras.layers import GRU, LSTM, Activation, Dense, Masking, Dropout, SimpleRNN, Input, Lambda, \
    Flatten, Reshape, MultiHeadAttention
from tensorflow.keras.layers import add, Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
from spektral.layers.convolutional.gcn_conv import GCNConv
import numpy as np

import tensorflow as tf


class GARG:

    def __init__(self):
        self.name = "garg"

    def build(self, step_size, location_input_dim, time_input_dim, num_users, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)

        location_category_input = Input((step_size,), dtype='int32', name='spatial')
        temporal_input = Input((step_size,), dtype='int32', name='temporal')
        country_input = Input((step_size,), dtype='int32', name='country')
        distance_input = Input((step_size,), dtype='int32', name='distance')
        duration_input = Input((step_size,), dtype='int32', name='duration')
        week_day_input = Input((step_size,), dtype='int32', name='week_day')
        user_id_input = Input((step_size,), dtype='int32', name='user')
        categories_distance_matrix = Input((location_input_dim, location_input_dim), dtype='float32', name='categories_distance_matrix')
        categories_temporal_matrix = Input((location_input_dim, 48), dtype='float32', name='categories_temporal_matrix')
        adjancency_matrix = Input((location_input_dim, location_input_dim), dtype='float32', name='adjacency_matrix')

        # The embedding layer converts integer encoded vectors to the specified
        # shape (none, input_lenght, output_dim) with random weights, which are
        # ajusted during the training turning helpful to find correlations between words.
        # Moreover, when you are working with one-hot-encoding
        # and the vocabulary is huge, you got a sparse matrix which is not computationally efficient.
        units = step_size*7
        emb1 = Embedding(input_dim=location_input_dim, output_dim=5, input_length=step_size)
        emb2 = Embedding(input_dim=48, output_dim=5, input_length=step_size)

        spatial_embedding = emb1(location_category_input)
        temporal_embedding = emb2(temporal_input)



        # Unlike LSTM, the GRU can find correlations between location/events
        # separated by longer times (bigger sentences)
        # spatial_embedding = Dropout(0.5)(spatial_embedding)
        # temporal_embedding = Dropout(0.5)(temporal_embedding)
        srnn = GRU(units, return_sequences=True)(spatial_embedding)
        srnn = Dropout(0.6)(srnn)
        #concat_1 = Concatenate()([srnn, temporal_embedding])

        att = MultiHeadAttention(key_dim=2,
                                 num_heads=1,
                                 name='Attention')(srnn, srnn)

        x = GCNConv(step_size)([categories_distance_matrix, adjancency_matrix])
        x = GCNConv(step_size)([x, adjancency_matrix])
        x = Flatten()(x)

        print("at", att.shape)
        #att = Concatenate()([srnn, att])
        att = Flatten()(att)
        print("att", att.shape)
        print("transposto", tf.transpose(att).shape)
        print("gc", x.shape)
        y_up = tf.matmul(att, x)
        #drop_1 = Dropout(0.6)(att)
        y_srnn = Dense(location_input_dim, activation='softmax')(y_up)



        model = Model(inputs=[location_category_input, temporal_input, country_input, distance_input, duration_input, week_day_input, user_id_input, adjancency_matrix, categories_distance_matrix, categories_temporal_matrix], outputs=[y_srnn], name="GARG_baseline")

        return model

