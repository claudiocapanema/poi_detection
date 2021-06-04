from tensorflow.keras.layers import GRU, LSTM, Activation, Dense, Masking, Dropout, SimpleRNN, Input, Lambda, \
    Flatten, Reshape, Embedding, Concatenate
from tensorflow.keras.models import Model
import numpy as np

import tensorflow as tf

class STF:

    def __init__(self):
        self.name = "stf"

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
        n = 2

        emb1 = Embedding(input_dim=location_input_dim, output_dim=3, input_length=step_size)
        emb2 = Embedding(input_dim=24, output_dim=3, input_length=step_size)

        spatial_embedding = emb1(location_category_input)
        temporal_embedding = emb2(temporal_input)

        concat_1 = Concatenate()([spatial_embedding, temporal_embedding])

        # Unlike LSTM, the GRU can find correlations between location/events
        # separated by longer times (bigger sentences)
        srnn = SimpleRNN(simple_rnn_units)(concat_1)
        drop_1 = Dropout(0.3)(srnn)
        y_srnn = Dense(location_input_dim, activation='softmax')(drop_1)

        model = Model(inputs=[location_category_input, temporal_input, country_input, distance_input, duration_input, week_day_input, user_id_input], outputs=[y_srnn], name="STF_RNN_baseline")

        return model