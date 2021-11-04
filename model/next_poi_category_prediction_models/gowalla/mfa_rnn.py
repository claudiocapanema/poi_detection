from tensorflow.keras.layers import GRU, LSTM, Activation, Dense, Masking, Dropout, SimpleRNN, Input, Lambda, \
    Flatten, Reshape, Concatenate, Embedding, MultiHeadAttention, ActivityRegularization, LayerNormalization, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping

from configuration.next_poi_category_prediciton_configuration import NextPoiCategoryPredictionConfiguration
from model.next_poi_category_prediction_models.neural_network_base_model import NNBase

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from spektral.layers.convolutional import ARMAConv, GCNConv, GATConv, DiffusionConv
from spektral.layers.pooling import GlobalAvgPool, GlobalAttentionPool, GlobalAttnSumPool

l2_reg = 5e-5           # L2 regularization rate
drop_out_rate = 0
patience = 3

# class AdaptativeGCN(Layer):
#     def __init__(self, main_channel,
#                  secondary_channel,
#                  kernel_initializer='glorot_uniform',
#                  kernel_regularizer=None,
#                  bias_initializer='zeros',
#                  bias_regularizer=None,
#                  bias_constraint=None,
#                  kernel_constraint=None):
#
#         super(AdaptativeGCN, self).__init__(dynamic=True)
#         self.main_channel = main_channel
#
#     def get_config(self):
#         config = {
#             # 'bias_regularizer': self.bias_regularizer,
#             # 'bias_constraint': self.bias_constraint,
#                 'main_layer': self.main_layer,
#                 'secondary_layer': self.secondary_layer,
#                 'main2_layer': self.main2_layer
#                 # 'main_use_bias': self.main_use_bias,
#                 # 'secondary_use_bias': self.secondary_use_bias,
#                 # 'main_activation': self.main_activation,
#                 # 'secondary_activation': self.secondary_activation,
#                 # 'bias_initializer': self.bias_initializer,
#                 # 'kernel_initializer': self.kernel_initializer,
#                 # 'kernel_regularizer': self.kernel_regularizer,
#                 # 'kernel_constraint': self.kernel_constraint,
#                 }
#         base_config = super().get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#     def build(self, input_shape):
#         assert len(input_shape) >= 2
#         input_dim = input_shape[0][-1]
#
#
#         self.v_bias_out = tf.Variable(1., trainable=True)
#         self.v_bias_out2 = tf.Variable(0., trainable=True)
#
#     def compute_output_shape(self, input_shape):
#         features_shape = input_shape[0]
#         output_shape = features_shape[:-1] + (self.main_channel,)
#         return output_shape
#
#     def call(self, inputs):
#
#
#
#         return output

class MFA_RNN(NNBase):

    def __init__(self):
        super().__init__("GRUenhaced original 10mil")

    def build(self, step_size, location_input_dim, time_input_dim, num_users, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)
        location_category_input = Input((step_size,), dtype='float32', name='spatial')
        temporal_input = Input((step_size,), dtype='float32', name='temporal')
        country_input = Input((step_size,), dtype='float32', name='country')
        distance_input = Input((step_size,), dtype='float32', name='distance')
        duration_input = Input((step_size,), dtype='float32', name='duration')
        week_day_input = Input((step_size,), dtype='float32', name='week_day')
        user_id_input = Input((step_size,), dtype='float32', name='user')
        pois_ids_input = Input((step_size,), dtype='float32', name='pois_ids')
        month_input = Input((step_size,), dtype='float32', name='month')
        categories_distance_matrix = Input((location_input_dim, location_input_dim), dtype='float32', name='categories_distance_matrix')
        categories_temporal_matrix = Input((location_input_dim, 48), dtype='float32', name='categories_temporal_matrix')
        adjancency_matrix = Input((location_input_dim, location_input_dim), dtype='float32', name='adjacency_matrix')
        categories_durations_matrix = Input((location_input_dim, location_input_dim), dtype='float32', name='categories_durations_matrix')
        poi_category_probabilities = Input((step_size, location_input_dim), dtype='float32', name='poi_category_probabilities')
        directed_adjancency_matrix = Input((location_input_dim, location_input_dim), dtype='float32', name='directed_adjacency_matrix')
        score = Input((1), dtype='float32', name='score')

        # adjancency_matrix = tf.cast(adjancency_matrix, dtype='float32')
        # categories_temporal_matrix = tf.cast(categories_temporal_matrix, dtype='float32')
        # categories_durations_matrix = tf.cast(categories_durations_matrix, dtype='float32')
        # categories_distance_matrix = tf.cast(categories_distance_matrix, dtype='float32')

        # The embedding layer converts integer encoded vectors to the specified
        # shape (none, input_lenght, output_dim) with random weights, which are
        # ajusted during the training turning helpful to find correlations between words.
        # Moreover, when you are working with one-hot-encoding
        # and the vocabulary is huge, you got a sparse matrix which is not computationally efficient.
        gru_units = 30
        emb_category = Embedding(input_dim=location_input_dim, output_dim=7, input_length=step_size)
        emb_time = Embedding(input_dim=time_input_dim, output_dim=3, input_length=step_size)
        emb_id = Embedding(input_dim=num_users, output_dim=2, input_length=step_size)
        emb_country = Embedding(input_dim=30, output_dim=2, input_length=step_size)
        emb_distance = Embedding(input_dim=51, output_dim=3, input_length=step_size)
        emb_duration = Embedding(input_dim=49, output_dim=3, input_length=step_size)
        emb_week_day = Embedding(input_dim=7, output_dim=3, input_length=step_size)
        emb_month = Embedding(input_dim=13, output_dim=3, input_length=step_size)

        locais = location_input_dim
        spatial_embedding = emb_category(location_category_input)
        temporal_embedding = emb_time(temporal_input)
        id_embedding = emb_id(user_id_input)
        country_embbeding = emb_country(country_input)
        distance_embbeding = emb_distance(distance_input)
        duration_embbeding = emb_duration(duration_input)
        week_day_embbeding = emb_week_day(week_day_input)
        month_embedding = emb_month(month_input)

        spatial_flatten = Flatten()(spatial_embedding)
        temporal_flatten = Flatten()(temporal_embedding)
        distance_flatten = Flatten()(distance_embbeding)
        duration_flatten = Flatten()(duration_embbeding)

        # melhor -
        distance_duration = tf.Variable(initial_value=0.1) * tf.multiply(distance_embbeding, duration_embbeding)

        l_p_flatten = Concatenate()([spatial_flatten, temporal_flatten, distance_flatten, duration_flatten])
        l_p = Concatenate()(
            [spatial_embedding, temporal_embedding, distance_embbeding, duration_embbeding, distance_duration])

        # l_p_flatten = Flatten()(l_p)
        # ids_flatten = Flatten()(id_flatten)

        # y_cup = tf.matmul(id_flatten, l_p_flatten)
        y_cup = Concatenate()([id_embedding, l_p])
        y_cup = Flatten()(y_cup)
        # y_cup = Dense(20)(y_cup)

        srnn = GRU(gru_units, return_sequences=True)(l_p)
        srnn = Dropout(0.5)(srnn)

        att = MultiHeadAttention(key_dim=2,
                                 num_heads=4,
                                 name='Attention')(srnn, srnn)

        distance_duration_matrix = tf.multiply(categories_distance_matrix, categories_durations_matrix)

        distance_matrix = categories_distance_matrix
        # distance_matrix = categories_distance_matrix
        # x_distances = self.graph_distances_a(distance_matrix, adjancency_matrix)

        adjancency_matrix_p = tf.matmul(poi_category_probabilities, adjancency_matrix)
        x_distances = GCNConv(22, activation='swish')([distance_matrix, adjancency_matrix])
        x_distances = Dropout(0.5)(x_distances)
        x_distances = GCNConv(10, activation='swish')([x_distances, adjancency_matrix])
        x_distances = Dropout(0.5)(x_distances)
        x_distances = Flatten()(x_distances)

        durations_matrix = categories_durations_matrix
        # durations_matrix = categories_durations_matrix
        # x_durations = self.graph_temporal_arma(durations_matrix, adjancency_matrix)
        x_durations = GCNConv(22, activation='swish')([durations_matrix, adjancency_matrix])
        # x_durations = Dropout(0.5)(x_durations)
        x_durations = GCNConv(10, activation='swish')([x_durations, adjancency_matrix])
        x_durations = Dropout(0.3)(x_durations)
        x_durations = Flatten()(x_durations)

        distance_duration_matrix = GCNConv(22, activation='swish')([distance_duration_matrix, adjancency_matrix])
        # x_durations = Dropout(0.5)(x_durations)
        distance_duration_matrix = GCNConv(10, activation='swish')([distance_duration_matrix, adjancency_matrix])
        distance_duration_matrix = Dropout(0.3)(distance_duration_matrix)
        distance_duration_matrix = Flatten()(distance_duration_matrix)

        print("at", att.shape)
        # att = Concatenate()([srnn, att])
        att = Flatten()(att)
        print("att", att.shape)
        print("transposto", tf.transpose(att).shape)
        print("gc", x_distances.shape)
        # y_up = tf.matmul(att, x)
        srnn = Flatten()(srnn)
        y_sup = Concatenate()([srnn, att, x_distances])
        y_sup = Dropout(0.3)(y_sup)
        y_sup = Dense(location_input_dim, activation='softmax')(y_sup)
        y_cup = Dropout(0.5)(y_cup)
        y_cup = Dense(location_input_dim, activation='softmax')(y_cup)
        spatial_flatten = Dense(location_input_dim, activation='softmax')(spatial_flatten)

        gnn = Concatenate()([x_durations, distance_duration_matrix])
        gnn = Dropout(0.3)(gnn)
        gnn = Dense(location_input_dim, activation='softmax')(gnn)

        # diagonal = tf.gather(directed_adjancency_matrix, location_input_dim, axis=1)
        # diagonal = tf.linalg.normalize(diagonal)[0]
        #diagonal = Flatten()(diagonal)
        # print("pi", poi_category_probabilities.shape)
        # print("cate", directed_adjancency_matrix.shape)
        # pc = Dense(location_input_dim, activation='relu')(poi_category_probabilities)
        # da = Dense(location_input_dim, activation='relu')(directed_adjancency_matrix)
        # diagonal = tf.matmul(pc, da)
        # diagonal = Flatten()(diagonal)
        # diagonal = Dense(location_input_dim, activation='softmax')(diagonal)
        # print("depois tensor: ", diagonal.shape)

        pc = Dense(14, activation='relu')(poi_category_probabilities)
        pc = Dropout(0.5)(pc)
        pc = Flatten()(pc)
        pc = Dense(location_input_dim, activation='softmax')(pc)

        r_entropy = tfp.distributions.Categorical(probs=y_sup).entropy()
        max_entropy = tf.reduce_mean(tfp.distributions.Categorical(probs=[1/location_input_dim]*location_input_dim).entropy())

        g_entropy = tfp.distributions.Categorical(probs=gnn).entropy()
        g_max_entropy = tf.reduce_mean(
            tfp.distributions.Categorical(probs=[1 / location_input_dim] * location_input_dim).entropy())

        y_up = tf.Variable(initial_value=1.) * y_cup + ((1/tf.math.reduce_mean(r_entropy)) + tf.Variable(0.5)) * tf.Variable(initial_value=1.) * y_sup + tf.Variable(
            initial_value=-0.2) * spatial_flatten + tf.Variable(initial_value=8.) * gnn
        model = Model(inputs=[location_category_input, temporal_input, country_input, distance_input, duration_input, week_day_input, user_id_input, pois_ids_input, month_input, adjancency_matrix, categories_distance_matrix, categories_temporal_matrix, categories_durations_matrix, score, poi_category_probabilities, directed_adjancency_matrix], outputs=[y_up], name="MFA-RNN")

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

    def graph_temporal_arma(self, x, adjacency):
        x = ARMAConv(22, iterations=1,
                     order=3,
                     share_weights=True,
                     dropout_rate=0,
                     activation='relu',
                     gcn_activation='relu',
                     kernel_regularizer=l2(l2_reg))([x, adjacency])

        x = Dropout(0.5)(x)

        x = ARMAConv(10, iterations=1,
                     order=1,
                     share_weights=True,
                     dropout_rate=drop_out_rate,
                     activation='relu',
                     gcn_activation='relu')([x, adjacency])
        x = Dropout(0.5)(x)

        return x

    def graph_distances_a(self, x, adjacency):
        x = ARMAConv(22, iterations=1,
                     order=2,
                     share_weights=True,
                     dropout_rate=0,
                     activation='swish',
                     gcn_activation='swish',
                     kernel_regularizer=l2(l2_reg))([x, adjacency])

        #x = Dropout(0.5)(x)

        x = ARMAConv(10, iterations=1,
                     order=2,
                     share_weights=True,
                     dropout_rate=drop_out_rate,
                     activation='swish',
                     gcn_activation='swish')([x, adjacency])
        x = Dropout(0.5)(x)

        return x

    def graph_durations_a(self, x, adjacency):
        x = ARMAConv(22, iterations=1,
                     order=3,
                     share_weights=True,
                     dropout_rate=0,
                     activation='swish',
                     gcn_activation='swish',
                     kernel_regularizer=l2(l2_reg))([x, adjacency])

        #x = Dropout(0.5)(x)

        x = ARMAConv(10, iterations=1,
                     order=2,
                     share_weights=True,
                     dropout_rate=drop_out_rate,
                     activation='swish',
                     gcn_activation='swish')([x, adjacency])
        x = Dropout(0.5)(x)

        return x

    def graph_temporal(self, x, adjacency):

        x = GCNConv(22, activation='relu')([x, adjacency])

        x = Dropout(0.5)(x)

        x = GCNConv(10, activation='relu')([x, adjacency])

        x = Dropout(0.5)(x)

        return x

    def graph_distances(self, x, adjacency):
        x = GCNConv(22, activation='relu')([x, adjacency])

        x = Dropout(0.5)(x)

        x = GCNConv(10, activation='relu')([x, adjacency])

        x = Dropout(0.5)(x)

        return x

    def graph_durations(self, x, adjacency):
        x = GCNConv(22, activation='relu')([x, adjacency])

        x = Dropout(0.5)(x)

        x = GCNConv(10, activation='relu')([x, adjacency])

        x = Dropout(0.5)(x)

        return x