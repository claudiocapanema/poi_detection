from tensorflow.keras.layers import GRU, LSTM, Activation, Dense, Masking, Dropout, SimpleRNN, Input, Lambda, \
    Flatten, Reshape, MultiHeadAttention
from tensorflow.keras.layers import add, Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
import numpy as np

import tensorflow as tf

class NEXT:

    def __init__(self):
        self.name = "next"

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
        emb1 = Embedding(input_dim=location_input_dim, output_dim=7, input_length=step_size)
        emb2 = Embedding(input_dim=48, output_dim=3, input_length=step_size)
        emb3 = Embedding(input_dim=num_users, output_dim=3, input_length=step_size)

        spatial_embedding = emb1(location_category_input)
        temporal_embedding = emb2(temporal_input)
        id_embbeding = emb3(user_id_input)



        # Unlike LSTM, the GRU can find correlations between location/events
        # separated by longer times (bigger sentences)
        # spatial_embedding = Dropout(0.5)(spatial_embedding)
        # temporal_embedding = Dropout(0.5)(temporal_embedding)
        concat_1 = Concatenate()([spatial_embedding, temporal_embedding])
        srnn = GRU(20, return_sequences=True)(concat_1)
        srnn = Dropout(0.5)(srnn)
        concat_2 = Concatenate()([srnn, id_embbeding])

        att = MultiHeadAttention(
            key_dim=1,
            num_heads=4,
            name='Attention')(concat_2, concat_2)

        print("aaaa")
        print(concat_2.shape)
        pos = self.positional_encoding(concat_2, 4)

        att = Concatenate()([att, pos])
        att = Flatten()(att)
        drop_1 = Dropout(0.4)(att)
        y_srnn = Dense(location_input_dim, activation='softmax')(drop_1)



        model = Model(inputs=[location_category_input, temporal_input, country_input, distance_input, duration_input, week_day_input, user_id_input], outputs=[y_srnn], name="NEXT_baseline")

        return model

    def positional_encoding(self, inputs,
                            maxlen,
                            masking=False,
                            scope="positional_encoding"):
        '''Sinusoidal Positional_Encoding. See 3.5
        inputs: 3d tensor. (N, T, E)
        maxlen: scalar. Must be >= T
        masking: Boolean. If True, padding positions are set to zeros.
        scope: Optional scope for `variable_scope`.
        returns
        3d tensor that has the same shape as inputs.
        '''

        E = inputs.shape[-1]  # static
        N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
                for pos in range(maxlen)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)

            # masks
            if masking:
                outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

            return tf.cast(outputs, tf.float32)