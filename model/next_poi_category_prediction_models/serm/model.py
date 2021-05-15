from tensorflow.keras.layers import GRU, LSTM, Activation, Dense, Masking, \
    Dropout, SimpleRNN, Input, Lambda, Flatten, Reshape
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
import numpy as np

class SERM:

    def __init__(self):
        self.name = "serm"

    def build(self, step_size, location_input_dim, time_input_dim, num_users, seed=None):
        if seed is not None:
            np.random.seed(seed)

        id_input = Input((step_size,), dtype='float32', name='id')
        s_input = Input((step_size,), dtype='int32', name='spatial')
        t_input = Input((step_size,), dtype='int32', name='temporal')

        # The embedding layer converts integer encoded vectors to the specified
        # shape (none, input_lenght, output_dim) with random weights, which are
        # ajusted during the training turning helpful to find correlations between words.
        # Moreover, when you are working with one-hot-encoding
        # and the vocabulary is huge, you got a sparse matrix which is not computationally efficient.
        units = 540
        n = 1
        id_output_dim = (units//8)*8 + 8*n - units
        emb3 = Embedding(input_dim=num_users, output_dim=50, input_length=step_size)
        emb1 = Embedding(input_dim=location_input_dim, output_dim=15, input_length=step_size)
        emb2 = Embedding(input_dim=48, output_dim=15, input_length=step_size)

        spatial_embedding = emb1(s_input)
        temporal_embedding = emb2(t_input)
        id_embedding = emb3(id_input)

        concat_1 = Concatenate()([spatial_embedding, temporal_embedding])
        print("concat_1: ", concat_1.shape)
        # Unlike LSTM, the GRU can find correlations between location/events
        # separated by longer times (bigger sentences)
        lstm_1 = LSTM(units, return_sequences=True)(concat_1)
        lstm_1 = Dropout(0.5)(lstm_1)
        lstm_1 = Dense(50)(lstm_1)
        print("lstm_1: ", lstm_1.shape, "id_embedding: ", id_embedding.shape)
        lstm_1 = Concatenate()([lstm_1, id_embedding])
        lstm_1 = Dropout(0.6)(lstm_1)
        #lstm_1 = concatenate(inputs=[lstm_1, id_embedding])
        #lstm_1 = Dropout(0.6)(lstm_1)
        flatten_1 = Flatten(name="ma_flatten_1")(lstm_1)
        dense_1 = Dense(location_input_dim)(flatten_1)
        print("dense_1: ", dense_1.shape)
        pred_location = Activation('softmax')(dense_1)

        model = Model(inputs=[s_input, t_input, id_input], outputs=[pred_location], name="serm")

        return model