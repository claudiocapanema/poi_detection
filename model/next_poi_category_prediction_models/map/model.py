from tensorflow.keras.layers import GRU, LSTM, Activation, Dense, Masking, Dropout, SimpleRNN, Input, Lambda, \
    Flatten, Reshape, MultiHeadAttention
from tensorflow.keras.layers import add, Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
import numpy as np


class MAP:

    def __init__(self):
        self.name = "map"

    def build(self, step_size, location_input_dim, time_input_dim, num_users, seed=None):
        if seed is not None:
            np.random.seed(seed)

        s_input = Input((step_size,), dtype='int32', name='spatial')
        t_input = Input((step_size,), dtype='int32', name='temporal')
        week_day_input = Input((step_size,), dtype='int32', name='daytype')
        id_input = Input((step_size,), dtype='int32', name='userid')

        # The embedding layer converts integer encoded vectors to the specified
        # shape (none, input_lenght, output_dim) with random weights, which are
        # ajusted during the training turning helpful to find correlations between words.
        # Moreover, when you are working with one-hot-encoding
        # and the vocabulary is huge, you got a sparse matrix which is not computationally efficient.
        simple_rnn_units = 15
        n = 2
        id_output_dim = (simple_rnn_units//8)*8 + 8*n - simple_rnn_units
        emb1 = Embedding(input_dim=location_input_dim, output_dim=5, input_length=step_size)
        emb2 = Embedding(input_dim=48, output_dim=5, input_length=step_size)

        spatial_embedding = emb1(s_input)
        temporal_embedding = emb2(t_input)



        # Unlike LSTM, the GRU can find correlations between location/events
        # separated by longer times (bigger sentences)
        # spatial_embedding = Dropout(0.5)(spatial_embedding)
        # temporal_embedding = Dropout(0.5)(temporal_embedding)
        srnn = SimpleRNN(300, return_sequences=True)(spatial_embedding)
        srnn = Dropout(0.5)(srnn)
        concat_1 = Concatenate(inputs=[srnn, temporal_embedding])

        att = MultiHeadAttention(num_heads=1,
                               name='Attention')(concat_1)

        att = Concatenate(inputs=[srnn, att])
        att = Flatten()(att)
        drop_1 = Dropout(0.6)(att)
        y_srnn = Dense(location_input_dim, activation='softmax')(drop_1)



        model = Model(inputs=[s_input, t_input, day_type, id_input], outputs=[y_srnn], name="MAP_baseline")

        return model

