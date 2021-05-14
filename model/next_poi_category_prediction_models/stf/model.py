from keras.layers import GRU, LSTM, Activation, Dense, Masking, Dropout, SimpleRNN, Input, Lambda, \
    Flatten, Reshape
from keras.layers.merge import add,concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
#from keras_multi_head import MultiHeadAttention
from keras.regularizers import l1, l2
from keras_self_attention import SeqSelfAttention
import numpy as np

class STF:

    def __init__(self):
        self.name = "stf"

    def build(self, step_size, location_input_dim, time_input_dim, num_users, seed=None):
        if seed is not None:
            np.random.seed(seed)

        s_input = Input((step_size,), dtype='int32', name='spatial')
        t_input = Input((step_size,), dtype='int32', name='temporal')
        id_input = Input((step_size,), dtype='float32', name='id')

        # The embedding layer converts integer encoded vectors to the specified
        # shape (none, input_lenght, output_dim) with random weights, which are
        # ajusted during the training turning helpful to find correlations between words.
        # Moreover, when you are working with one-hot-encoding
        # and the vocabulary is huge, you got a sparse matrix which is not computationally efficient.
        simple_rnn_units = 560
        n = 2

        emb1 = Embedding(input_dim=location_input_dim, output_dim=15, input_length=step_size)
        emb2 = Embedding(input_dim=48, output_dim=15, input_length=step_size)
        emb3 = Embedding(input_dim=num_users, output_dim=15, input_length=step_size)

        spatial_embedding = emb1(s_input)
        temporal_embedding = emb2(t_input)
        id_embedding = emb3(id_input)

        concat_1 = concatenate(inputs=[spatial_embedding, temporal_embedding])
        # concat_1 = Dropout(0.5)(concat_1)
        # print("concat_1: ", concat_1.shape)

        # Unlike LSTM, the GRU can find correlations between location/events
        # separated by longer times (bigger sentences)
        srnn = SimpleRNN(simple_rnn_units)(concat_1)
        drop_1 = Dropout(0.5)(srnn)
        y_srnn = Dense(location_input_dim, activation='softmax')(drop_1)

        model = Model(inputs=[s_input, t_input, id_input], outputs=[y_srnn], name="STF_RNN_baseline")

        return model