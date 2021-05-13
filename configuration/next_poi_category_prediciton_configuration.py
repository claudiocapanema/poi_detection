from enum import Enum
import pytz
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Nadam


class NextPoiCategoryPredictionConfiguration:

    # Radius for the nearestneighbors algorithm - 100m
    def __init__(self):
        self.SEQUENCES_SIZE = ("sequences_size", 8)

        self.N_SPLITS = ("n_splits", 2)

        self.EPOCHS = ("epochs", 10)

        self.N_REPLICATIONS = ("n_replications", 1)

        self.BATCH = ("batch", {'serm': 200, 'map': 200, 'stf': 200})

        self.OPTIMIZER = ("learning_rate", {'serm': Adam(), 'map': Adam(), 'stf': Adam(epsilon=0.1, clipnorm=1)})

        self.OUTPUT_BASE_DIR = (
        "output_dir", "output/poi_categorization_sequential_baselines/", False, "output directory for the poi_categorization")

        self.MODEL_NAME = ("model_name", {'serm': "serm/", 'map': "map/", 'stf': "stf/"})

        self.DATASET_TYPE = ("dataset_type", {'foursquare': "foursquare/", 'weeplaces': "weeplaces/"})

        self.CATEGORY_TYPE = ("category_type",
                         {'osm': "13_categories/",
                          'reduced_osm': "9_categories/", '7_categories_osm': "7_categories/"})

        self.CLASS_WEIGHT = ("class_weight",
                        {'7_categories_osm': {'serm': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
                                              'map': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
                                              'stf': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}}})

        self.DATASET_COLUMNS = ("dataset_columns", {"users_steps": {"datetime": "datetime",
                                                                  "userid": "id",
                                                                  "locationid": "placeid",
                                                                  "category": "category",
                                                                  "latitude": "latitude",
                                                                  "longitude": "longitude"}})

        self.CATEGORIES = ['home', 'work', 'other', 'displacement', 'amenity', 'leisure', 'office', 'shop', 'sport', 'tourism']

        self.CATEGORIES_TO_INT = ("categories_to_int", {"users_steps":
                                                            {"10_categories": {self.CATEGORIES[i]: i for i in range(len(self.CATEGORIES))}}})

        self.MAX_POIS = ("max_pois", 10)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_key(self):
        return self.value[0]

    def get_value(self):
        return self.value[1]