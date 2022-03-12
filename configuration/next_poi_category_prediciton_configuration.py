from enum import Enum
import pytz
from tensorflow.keras.optimizers import Adam, Adadelta, SGD, RMSprop, Nadam, Ftrl
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.losses import CategoricalCrossentropy, CategoricalHinge


class NextPoiCategoryPredictionConfiguration:

    # Radius for the nearestneighbors algorithm - 100m
    def __init__(self):
        # 7
        self.SEQUENCES_SIZE = ("sequences_size", {'users_steps': 3, 'gowalla': 3})

        self.N_SPLITS = ("n_splits", 5)

        self.EPOCHS = ("epochs", {'users_steps': {'mfa': 10, 'serm': 10, 'map': 10, 'stf': 10, 'next': 10, 'garg': 10},
                                'gowalla': {'mfa': 35, 'serm': 25, 'map': 25, 'stf': 25, 'next': 11, 'garg': 25}})

        self.N_REPLICATIONS = ("n_replications", 1)

        self.BATCH = ("batch", {'users_steps': {'mfa': 350, 'serm': 400, 'map': 400, 'stf': 400, 'next': 300, 'garg': 400},
                                'gowalla': {'mfa': 400, 'serm': 400, 'map': 400, 'stf': 400, 'next': 400, 'garg': 400}})

        self.OPTIMIZER = ("learning_rate", {'users_steps': {'mfa': Adam(), 'serm': Adam(), 'map': Adam(), 'stf': Adam(),
                                            'next': Adam(), 'garg': Adam()},
                                            'gowalla': {'mfa': Adam(learning_rate=0.001, beta_1=0.8, beta_2=0.9), 'serm': Adam(learning_rate=0.001, beta_1=0.8, beta_2=0.9), 'map': Adam(learning_rate=0.001, beta_1=0.8, beta_2=0.9), 'stf': Adam(learning_rate=0.001, beta_1=0.8, beta_2=0.9),
                                            'next': Adam(learning_rate=0.0007, beta_1=0.8, beta_2=0.9), 'garg': Adam(learning_rate=0.001, beta_1=0.8, beta_2=0.9)}})

        self.LOSS = ("learning_rate", {'mfa': CategoricalCrossentropy(), 'serm': CategoricalCrossentropy(), 'map': CategoricalCrossentropy(),
                                            'stf': CategoricalCrossentropy(),
                                            'next': CategoricalCrossentropy(), 'garg': CategoricalCrossentropy()})

        self.FORMAT_MODEL_NAME = ("format_model_name", {'mfa': 'POI-RGNN', 'stf': 'STF-RNN', 'map': 'MAP', 'serm': 'SERM', 'next': 'MHA+PE', 'garg': 'GARG'})

        self.OUTPUT_BASE_DIR = (
        "output_dir", "output/next_poi_category_prediction/", False, "output directory for the poi_categorization")

        self.MODEL_NAME = ("model_name", {'mfa': "mfa/", 'serm': "serm/", 'map': "map/", 'stf': "stf/",
                                          'next': "next/", 'garg': "garg/"})

        self.DATASET_TYPE = ("dataset_type", {'users_steps': "users_steps/", 'gowalla': 'gowalla/'})

        self.CATEGORY_TYPE = ("category_type",
                         {'10_categories': "10_categories/",
                          '8_categories': "8_categories/",
                          '7_categories': "7_categories/",
                          '3_categories': "3_categories/"})

        self.CLASS_WEIGHT = ("class_weight",
                        {'10_categories': {'serm': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 3, 7: 1, 8: 1, 9: 3},
                                           'map': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 3, 7: 1, 8: 1, 9: 3},
                                           'stf': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 3, 7: 1, 8: 1, 9: 3},
                                           'mfa': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 3, 7: 1, 8: 1, 9: 3},
                                           'next': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 3, 7: 1, 8: 1, 9: 3},
                                           'garg': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 3, 7: 1, 8: 1, 9: 3}},
                         '8_categories': {'serm': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
                                           'map': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
                                           'stf': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
                                           'mfa': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
                                          'next': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
                                          'garg': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}},
                         '7_categories': {'serm': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
                                          'map': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
                                          'stf': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
                                          'mfa': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
                                          'next': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
                                          'garg': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}},
                         '3_categories': {'serm': {0: 1, 1: 1, 2: 1},
                                           'map': {0: 1, 1: 1, 2: 1},
                                           'stf': {0: 1, 1: 1, 2: 1},
                                           'mfa': {0: 1, 1: 1, 2: 1},
                                          'next': {0: 1, 1: 1, 2: 1}}})

        self.DATASET_COLUMNS = ("dataset_columns", {"users_steps": {"datetime": "datetime",
                                                                  "userid": "id",
                                                                  "locationid": "poi_id",
                                                                  "category": "poi_resulting",
                                                                  "latitude": "latitude",
                                                                  "longitude": "longitude",
                                                                    "country": "country_name",
                                                                    "state": "state_name"},
                                                    "gowalla": {"datetime": "local_datetime",
                                                                    "userid": "userid",
                                                                    "locationid": "placeid",
                                                                    "category": "category",
                                                                    "latitude": "latitude",
                                                                    "longitude": "longitude",
                                                                    "country": "country_name",
                                                                    "state": "county_name"
                                                                    }})

        self.CATEGORIES_10 = ['Home', 'Work', 'Other', 'Shopping', 'Community', 'Food', 'Entertainment', 'Travel', 'Outdoors', 'Nightlife']

        self.CATEGORIES_8 = ['Home', 'Work', 'Other', 'Commuting', 'Amenity', 'Leisure', 'Shop', 'Tourism']

        self.GOWALLA_7_CATEGORIES = ['Shopping', 'Community', 'Food', 'Entertainment', 'Travel', 'Outdoors', 'Nightlife']

        self.CATEGORIES_3 = ['displacement', 'home', 'other']

        self.CATEGORIES_TO_INT = ("categories_to_int", {"users_steps":
                                                            {"10_categories": {self.CATEGORIES_10[i]: i for i in range(len(self.CATEGORIES_10))},
                                                             "8_categories": {self.CATEGORIES_8[i]: i for i in range(len(self.CATEGORIES_8))},
                                                             "3_categories": {self.CATEGORIES_3[i]: i for i in range(len(self.CATEGORIES_3))}},
                                                        "gowalla":
                                                            {"7_categories": {self.GOWALLA_7_CATEGORIES[i]: i for i in range(len(self.GOWALLA_7_CATEGORIES))}}})

        self.INT_TO_CATEGORIES = ("int_to_categories", {"users_steps": {"10_categories": {str(i): self.CATEGORIES_10[i] for i in range(len(self.CATEGORIES_10))},
                                                                        "8_categories": {str(i): self.CATEGORIES_8[i] for i in range(len(self.CATEGORIES_8))},
                                                                        "3_categories": {str(i): self.CATEGORIES_3[i] for i in range(len(self.CATEGORIES_3))}},
                                                        "gowalla": {"7_categories": {str(i): self.GOWALLA_7_CATEGORIES[i] for i in range(len(self.GOWALLA_7_CATEGORIES))}}})

        self.MAX_POIS = ("max_pois", 10)

        self.REPORT_10_INT_CATEGORIES = ("report_10_int_categories",
                                        {'0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '3': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '4': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '5': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '6': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '7': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '8': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '9': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         'accuracy': [],
                                         'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         'weighted avg': {'precision': [], 'recall': [], 'f1-score': [],
                                                          'support': []}},
                                        "report")

        self.REPORT_7_INT_CATEGORIES = ("report_7_int_categories",
                                        {'0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '3': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '4': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '5': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '6': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         'accuracy': [],
                                         'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         'weighted avg': {'precision': [], 'recall': [], 'f1-score': [],
                                                          'support': []}},
                                        "report")

        self.REPORT_8_INT_CATEGORIES = ("report_8_int_categories",
                                         {'0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '3': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '4': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '5': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '6': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '7': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          'accuracy': [],
                                          'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          'weighted avg': {'precision': [], 'recall': [], 'f1-score': [],
                                                           'support': []}},
                                         "report")

        self.REPORT_3_INT_CATEGORIES = ("report_10_int_categories",
                                         {'0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          'accuracy': [],
                                          'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          'weighted avg': {'precision': [], 'recall': [], 'f1-score': [],
                                                           'support': []}},
                                         "report")

        self.REPORT_MODEL = ("report_model",
                             {'10_categories': self.REPORT_10_INT_CATEGORIES[1],
                              '8_categories': self.REPORT_8_INT_CATEGORIES[1],
                              '7_categories': self.REPORT_7_INT_CATEGORIES[1],
                              '3_categories': self.REPORT_3_INT_CATEGORIES[1]})

        self.NUMBER_OF_CATEGORIES = ("number_of_categories", {'10_categories': 10,
                                                              '8_categories': 8,
                                                              '7_categories': 7,
                                                              '3_categories': 3})

        self.STEP_SIZE = ("step_size", 8)

        self.DISTANCE_SIGMA = ("distance_sigma", {'users_steps': 5,
                                                  'gowalla': 10})

        self.DURATION_SIGMA = ("duration_sigma", {'users_steps': 5,
                                                  'gowalla': 10})

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_key(self):
        return self.value[0]

    def get_value(self):
        return self.value[1]