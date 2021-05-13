from configuration.next_poi_category_prediciton_configuration import NextPoiCategoryPredictionConfiguration
from domain.next_poi_category_prediction_sequences_generation_domain import NextPoiCategoryPredictionSequencesGenerationDomain
from configuration.next_poi_category_prediction_sequences_generation_confiiguration import SequencesGenerationForPoiCategorizationSequentialBaselinesConfiguration

from foundation.configuration.input import Input

class NextPoiCategoryPredictionSequencesGenerationJob:

    def __init__(self):
        self.poi_categorization_configuration = NextPoiCategoryPredictionConfiguration()
        self.sequences_generation_for_poi_categorization_sequential_baselines_domain = NextPoiCategoryPredictionSequencesGenerationDomain(Input.get_instance().inputs['dataset_name'])

    def start(self):
        users_checkin_filename = Input.get_instance().inputs['users_steps_filename']
        dataset_name = Input.get_instance().inputs['dataset_name']
        categories_type = Input.get_instance().inputs['categories_type']
        users_sequences_folder = Input.get_instance().inputs['users_sequences_folder']
        print("Dataset: ", Input.get_instance().inputs['dataset_name'])

        userid_column = self.poi_categorization_configuration.DATASET_COLUMNS[1][dataset_name]['userid']
        category_column = self.poi_categorization_configuration.DATASET_COLUMNS[1][dataset_name]['category']
        locationid_column  = self.poi_categorization_configuration.DATASET_COLUMNS[1][dataset_name]['locationid']
        datetime_column = self.poi_categorization_configuration.DATASET_COLUMNS[1][dataset_name]['datetime']
        categories_to_int_osm = self.poi_categorization_configuration.CATEGORIES_TO_INT[1][dataset_name][categories_type]
        max_pois = self.poi_categorization_configuration.MAX_POIS[1]
        sequences_size = SequencesGenerationForPoiCategorizationSequentialBaselinesConfiguration.SEQUENCES_SIZE.get_value()

        users_checkin = self.sequences_generation_for_poi_categorization_sequential_baselines_domain.read_csv(users_checkin_filename, datetime_column)

        users_sequences = self.sequences_generation_for_poi_categorization_sequential_baselines_domain.generate_sequences(users_checkin,
                                                                                                                          sequences_size,
                                                                                                                          max_pois,
                                                                                                                          userid_column,
                                                                                                                          category_column,
                                                                                                                          locationid_column,
                                                                                                                          datetime_column,
                                                                                                                          categories_to_int_osm)

        self.sequences_generation_for_poi_categorization_sequential_baselines_domain.sequences_to_csv(users_sequences, users_sequences_folder, dataset_name, categories_type)


