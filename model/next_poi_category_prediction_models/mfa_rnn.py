from tensorflow.keras.layers import GRU, LSTM, Activation, Dense, Masking, Dropout, SimpleRNN, Input, Lambda, \
    Flatten, Reshape, Concatenate, Embedding, MultiHeadAttention, ActivityRegularization, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2

from configuration.next_poi_category_prediciton_configuration import NextPoiCategoryPredictionConfiguration
from model.next_poi_category_prediction_models.neural_network_base_model import NNBase

import numpy as np
import tensorflow as tf

class MFA_RNN(NNBase):

    def __init__(self):
        super().__init__("GRUenhaced original 10mil")

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
        gru_units = 330
        emb_category = Embedding(input_dim=location_input_dim, output_dim=7, input_length=step_size)
        emb_time = Embedding(input_dim=time_input_dim, output_dim=5, input_length=step_size)
        emb_id = Embedding(input_dim=num_users, output_dim=5, input_length=step_size)
        emb_country = Embedding(input_dim=30, output_dim=2, input_length=step_size)
        emb_distance = Embedding(input_dim=51, output_dim=5, input_length=step_size)
        emb_duration = Embedding(input_dim=25, output_dim=5, input_length=step_size)
        emb_week_day = Embedding(input_dim=7, output_dim=5, input_length=step_size)

        spatial_embedding = emb_category(location_category_input)
        temporal_embedding = emb_time(temporal_input)
        id_embedding = emb_id(user_id_input)
        country_embbeding = emb_country(country_input)
        distance_embbeding = emb_distance(distance_input)
        duration_embbeding = emb_duration(duration_input)
        week_day_embbeding = emb_week_day(week_day_input)

        concat_1 = Concatenate()([spatial_embedding, temporal_embedding, distance_embbeding, duration_embbeding])

        # Unlike LSTM, the GRU can find correlations between location/events
        # separated by longer times (bigger sentences)
        #drop_1 = Dropout(0.5)(concat_1)
        gru_1 = LSTM(gru_units, return_sequences=True)(concat_1)
        print("gru_1: ", gru_1.shape, "id_embedding: ", id_embedding.shape)

        #concat_2 = concatenate(inputs=[gru_1, id_embedding])
        #concat_2 = concatenate(inputs=[concat_2, daytype_embedding])
        #concat_2 = Dropout(0.5)(concat_2)
        #print("concat_2: ", concat_2.shape)
        # , activation='elu', kernel_regularizer=tf.keras.regularizers.L2()
        gru_1 = Dense(150)(gru_1)
        #gru_1 = Dropout(0.4)(gru_1)
        y_mhsa = self.mhsa(input=gru_1)

        final = Concatenate()([y_mhsa, gru_1])
        final = Concatenate()([final, id_embedding])
        final = Concatenate()([final, country_embbeding])
        final = Flatten()(final)
        final = Dropout(0.4)(final)
        final = Dense(location_input_dim)(final)
        final = Activation('softmax', name='ma_activation_1')(final)

        model = Model(inputs=[location_category_input, temporal_input, country_input, distance_input, duration_input, week_day_input, user_id_input], outputs=[final], name="MFA-RNN")

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
