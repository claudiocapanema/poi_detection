from tensorflow.keras.layers import GRU, LSTM, Activation, Dense, Masking, \
    Dropout, SimpleRNN, Input, Lambda, Flatten, Reshape
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
import numpy as np

import tensorflow as tf

class SERMUsersSteps:

    def __init__(self):
        self.name = "serm"

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
        units = 50

        emb3 = Embedding(input_dim=num_users, output_dim=3, input_length=step_size)
        emb1 = Embedding(input_dim=location_input_dim, output_dim=7, input_length=step_size)
        emb2 = Embedding(input_dim=48, output_dim=3, input_length=step_size)

        spatial_embedding = emb1(location_category_input)
        temporal_embedding = emb2(temporal_input)
        id_embedding = emb3(user_id_input)

        concat_1 = Concatenate()([spatial_embedding, temporal_embedding])
        # Unlike LSTM, the GRU can find correlations between location/events
        # separated by longer times (bigger sentences)
        lstm_1 = LSTM(units, return_sequences=True)(concat_1)
        # lstm_1 = Dropout(0.5)(lstm_1)
        # lstm_1 = Dense(24)(lstm_1)
        lstm_1 = Concatenate()([lstm_1, id_embedding])
        flatten_1 = Flatten(name="ma_flatten_1")(lstm_1)
        flatten_1 = Dropout(0.5)(flatten_1)
        dense_1 = Dense(location_input_dim)(flatten_1)
        pred_location = Activation('softmax')(dense_1)

        model = Model(inputs=[location_category_input, temporal_input, country_input, distance_input, duration_input, week_day_input, user_id_input], outputs=[pred_location], name="serm")

        return model