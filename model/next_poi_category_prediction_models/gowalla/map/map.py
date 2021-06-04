from tensorflow.keras.layers import GRU, LSTM, Activation, Dense, Masking, Dropout, SimpleRNN, Input, Lambda, \
    Flatten, Reshape, MultiHeadAttention
from tensorflow.keras.layers import add, Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
import numpy as np

import tensorflow as tf


class MAP:

    def __init__(self):
        self.name = "map"

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

        # The embedding layer converts integer encoded vectors to the specified
        # shape (none, input_lenght, output_dim) with random weights, which are
        # ajusted during the training turning helpful to find correlations between words.
        # Moreover, when you are working with one-hot-encoding
        # and the vocabulary is huge, you got a sparse matrix which is not computationally efficient.
        simple_rnn_units = 10
        emb1 = Embedding(input_dim=location_input_dim, output_dim=3, input_length=step_size)
        emb2 = Embedding(input_dim=48, output_dim=3, input_length=step_size)

        spatial_embedding = emb1(location_category_input)
        temporal_embedding = emb2(temporal_input)



        # Unlike LSTM, the GRU can find correlations between location/events
        # separated by longer times (bigger sentences)
        # spatial_embedding = Dropout(0.5)(spatial_embedding)
        # temporal_embedding = Dropout(0.5)(temporal_embedding)
        srnn = SimpleRNN(20, return_sequences=True)(spatial_embedding)
        srnn = Dropout(0.4)(srnn)
        concat_1 = Concatenate()([srnn, temporal_embedding])

        att = MultiHeadAttention(key_dim=2,
                                 num_heads=1,
                                 name='Attention')(concat_1, concat_1)

        att = Concatenate()([srnn, att])
        att = Flatten()(att)
        drop_1 = Dropout(0.4)(att)
        y_srnn = Dense(location_input_dim, activation='softmax')(drop_1)



        model = Model(inputs=[location_category_input, temporal_input, country_input, distance_input, duration_input, week_day_input, user_id_input], outputs=[y_srnn], name="MAP_baseline")

        return model

