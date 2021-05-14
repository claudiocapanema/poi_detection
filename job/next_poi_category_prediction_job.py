from configuration.next_poi_category_prediciton_configuration import NextPoiCategoryPredictionConfiguration
from domain.next_poi_category_prediction_domain import NextPoiCategoryPredictionDomain
from loader.next_poi_category_prediction_loader import NextPoiCategoryPredictionLoader
from foundation.configuration.input import Input

class NextPoiCategoryPredictionJob:

    def __init__(self):
        self.next_poi_category_prediction_configuration = NextPoiCategoryPredictionConfiguration()
        self.next_poi_category_prediction_domain = NextPoiCategoryPredictionDomain(Input.get_instance().inputs['dataset_name'])
        self.next_poi_category_prediction_loader = NextPoiCategoryPredictionLoader()

    def start(self):
        dataset_name = Input.get_instance().inputs['dataset_name']
        categories_type = Input.get_instance().inputs['categories_type']
        users_sequences_filename = Input.get_instance().inputs['users_sequences']
        model_name = Input.get_instance().inputs['baseline']

        sequences_size = self.next_poi_category_prediction_configuration.SEQUENCES_SIZE[1]
        n_splits = self.next_poi_category_prediction_configuration.N_SPLITS[1]
        epochs = self.next_poi_category_prediction_configuration.EPOCHS[1]
        n_replications = self.next_poi_category_prediction_configuration.N_REPLICATIONS[1]
        batch = self.next_poi_category_prediction_configuration.BATCH[1][model_name]

        output_base_dir = self.next_poi_category_prediction_configuration.OUTPUT_BASE_DIR[1]
        dataset_type_dir = self.next_poi_category_prediction_configuration.DATASET_TYPE[1][dataset_name]
        category_type_dir = self.next_poi_category_prediction_configuration.CATEGORY_TYPE[1][categories_type]
        model_name_dir = self.next_poi_category_prediction_configuration.MODEL_NAME[1][model_name]
        class_weight = self.next_poi_category_prediction_configuration.CLASS_WEIGHT[1][categories_type][model_name]
        optimizer = self.next_poi_category_prediction_configuration.OPTIMIZER[1][model_name]
        output_dir = self.next_poi_category_prediction_domain.\
            output_dir(output_base_dir, dataset_type_dir, category_type_dir, model_name_dir)
        report_model = self.next_poi_category_prediction_configuration.REPORT_MODEL[1][categories_type]
        number_of_categories = self.next_poi_category_prediction_configuration.NUMBER_OF_CATEGORIES[1][categories_type]

        filename=0

        users_trajectories, users_train_indexes, users_test_indexes, num_users = self.next_poi_category_prediction_domain.read_sequences(users_sequences_filename, n_splits)

        num_users +=1
        print("numero usuarios: ", num_users)
        output = output_dir + str(n_splits) + "_folds/" + str(n_replications) + "_replications/"
        folds_histories, base_report = self.next_poi_category_prediction_domain.\
            run_tests_one_location_output_k_fold(users_trajectories,
                                                users_train_indexes,
                                                users_test_indexes,
                                                n_replications,
                                                n_splits,
                                                model_name,
                                                epochs,
                                                class_weight,
                                                filename,
                                                sequences_size,
                                                report_model,
                                                 number_of_categories,
                                                 batch,
                                                 num_users,
                                                 optimizer,
                                                 output
                                                )

        print("numero classes: ", number_of_categories)
        self.next_poi_category_prediction_loader.plot_history_metrics(folds_histories, base_report, output_dir, n_splits, n_replications)
        self.next_poi_category_prediction_loader.save_report_to_csv(output_dir, base_report, n_splits, n_replications)



