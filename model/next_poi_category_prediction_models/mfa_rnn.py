from tensorflow.keras.layers import GRU, LSTM, Activation, Dense, Masking, Dropout, SimpleRNN, Input, Lambda, \
    Flatten, Reshape, Concatenate, Embedding, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2

from configuration.next_poi_category_prediciton_configuration import NextPoiCategoryPredictionConfiguration
from model.next_poi_category_prediction_models.neural_network_base_model import NNBase

class MFA_RNN(NNBase):

    def __init__(self):
        super().__init__("GRUenhaced original 10mil")

    def build(self, step_size, location_input_dim, time_input_dim, num_users, seed=None):
        location_category_input = Input((step_size,), dtype='int32', name='spatial')
        temporal_input = Input((step_size,), dtype='int32', name='temporal')
        user_id_input = Input((step_size,), dtype='float32', name='id')
        country_input = Input((step_size,), dtype='float32', name='daytype')

        # The embedding layer converts integer encoded vectors to the specified
        # shape (none, input_lenght, output_dim) with random weights, which are
        # ajusted during the training turning helpful to find correlations between words.
        # Moreover, when you are working with one-hot-encoding
        # and the vocabulary is huge, you got a sparse matrix which is not computationally efficient.
        gru_units = 30
        n = 2
        id_output_dim = (gru_units//8)*8 + 8*n - gru_units
        emb1 = Embedding(input_dim=location_input_dim, output_dim=5, input_length=step_size)
        emb2 = Embedding(input_dim=time_input_dim, output_dim=10, input_length=step_size)
        emb3 = Embedding(input_dim=num_users, output_dim=2, input_length=step_size)
        emb4 = Embedding(input_dim=30, output_dim=1, input_length=step_size)

        spatial_embedding = emb1(location_category_input)
        temporal_embedding = emb2(temporal_input)
        id_embedding = emb3(user_id_input)
        country_embbeding = emb4(country_input)

        concat_1 = Concatenate()([spatial_embedding, temporal_embedding])
        concat_1 = Concatenate()([concat_1, country_embbeding])
        print("concat_1: ", concat_1.shape)

        # Unlike LSTM, the GRU can find correlations between location/events
        # separated by longer times (bigger sentences)
        drop_1 = Dropout(0.5)(concat_1)
        gru_1 = GRU(gru_units, return_sequences=True)(drop_1)
        print("gru_1: ", gru_1.shape, "id_embedding: ", id_embedding.shape)
        gru_1 = Dropout(0.4)(gru_1)
        #concat_2 = concatenate(inputs=[gru_1, id_embedding])
        #concat_2 = concatenate(inputs=[concat_2, daytype_embedding])
        #concat_2 = Dropout(0.5)(concat_2)
        #print("concat_2: ", concat_2.shape)

        y_mhsa = self.mhsa(input=gru_1, id_embedding=country_embbeding)
        #y_pe = self.dense_output(input=gru_1, id_embedding=id_embedding)

        #y_pe = self.pe(input=concat_2, id_embedding=daytype_embedding)
        # y_pe = concatenate(inputs=[y_pe, daytype_embedding])
        #id_embedding = Flatten()(id_embedding)
        final = Concatenate()([y_mhsa, gru_1])
        final = Concatenate()([final, id_embedding])
        final = Flatten()(final)
        final = Dropout(0.2)(final)
        final = Dense(location_input_dim)(final)
        final = Activation('softmax', name='ma_activation_1')(final)

        model = Model(inputs=[location_category_input, temporal_input, country_input, user_id_input], outputs=[final], name="MFA-RNN")

        return model

    def mhsa(self, input, id_embedding, numLocations=3):
        # att_layer = MultiHeadAttention(
        #     head_num=8,
        #     name='Multi-Head',
        # )(concat_2)
        att_layer = MultiHeadAttention(
            key_dim=2,
            num_heads=4,
            name='Multi-Head-self-attention',
        )(input, input)
        print("att", att_layer.shape, "att id_embedding", id_embedding.shape)

        return att_layer

    def pe(self, input, id_embedding, numTimeslots=3):
        # att_layer = MultiHeadAttention(
        #     head_num=8,
        #     name='Multi-Head',
        # )(concat_2)

        pe_layer = AddPositionalEncoding()(input)
        print("pe_layer: ", pe_layer.shape)

        # concat_3 = concatenate(inputs=[pe_layer, id_embedding])
        # print("pe concat_3: ", concat_3.shape)

        # concat_3 = Reshape((reshape_size,), input_shape=(4,))(concat_3)
        # print("pe concat_3:  ", concat_3.shape, concat_3)
        flatten_1 = Flatten(name="pe_flatten_1")(pe_layer)

        drop_1 = Dropout(0.5, name="pe_drop_1")(flatten_1)

        # dense_1 = Dense(numTimeslots, kernel_regularizer=l2(0.01), name='pe_dense_1')(drop_1)
        # y_pe = Activation('softmax', name='pe_activation_1')(dense_1)

        return drop_1

    def dense_output(self, input, id_embedding, numLocations=3):
        #concat_3 = concatenate(inputs=[input, id_embedding])
        dense_1 = Dense(5, name='dense_d')(input)
        #flatten_1 = Flatten(name="d_flatten_1")(dense_1)
        #dense_1 = Dropout(0.5)(dense_1)


        return dense_1