import numpy as np
import pandas as pd
from pathlib import Path
import os
from contextlib import suppress
import statistics as st

from domain.user_step_domain import UserStepDomain
from foundation.abs_classes.job import Job
from configuration.next_poi_category_prediciton_configuration import NextPoiCategoryPredictionConfiguration
from domain.next_poi_category_prediction_domain import NextPoiCategoryPredictionDomain
from domain.points_of_interest_domain import PointsOfInterestDomain
from loader.file_loader import FileLoader
from foundation.configuration.input import Input
from extractor.file_extractor import FileExtractor

class PerformancePlots(Job):

    def __init__(self):
        self.user_step_domain = UserStepDomain()
        self.points_of_interest_domain = PointsOfInterestDomain()
        self.next_poi_category_prediction_configuration = NextPoiCategoryPredictionConfiguration()
        self.next_poi_category_prediction_domain = NextPoiCategoryPredictionDomain(
            Input.get_instance().inputs['dataset_name'])
        self.file_loader = FileLoader()
        self.file_extractor = FileExtractor()

    def start(self):
        dataset_name = Input.get_instance().inputs['dataset_name']
        categories_type = Input.get_instance().inputs['categories_type']

        n_splits = self.next_poi_category_prediction_configuration.N_SPLITS[1]
        n_replications = self.next_poi_category_prediction_configuration.N_REPLICATIONS[1]
        output_base_dir = self.next_poi_category_prediction_configuration.OUTPUT_BASE_DIR[1]
        dataset_type_dir = self.next_poi_category_prediction_configuration.DATASET_TYPE[1][dataset_name]
        category_type_dir = self.next_poi_category_prediction_configuration.CATEGORY_TYPE[1][categories_type]

        model_report = {'mfa': {}, 'stf': {}}
        for model_name in model_report.keys():
            model_name_dir = self.next_poi_category_prediction_configuration.MODEL_NAME[1][model_name]
            output_dir = self.next_poi_category_prediction_domain. \
                output_dir(output_base_dir, dataset_type_dir, category_type_dir, model_name_dir)
            output = output_dir + str(n_splits) + "_folds/" + str(n_replications) + "_replications/"


            model_report[model_name]['precision'] = self.file_extractor.read_csv(output + "precision.csv")
            model_report[model_name]['recall'] = self.file_extractor.read_csv(output + "recall.csv")
            model_report[model_name]['fscore'] = self.file_extractor.read_csv(output + "fscore.csv")

        print(model_report)
        columns = ['home','work', 'other', 'displacement', 'amenity', 'leisure', 'shop', 'tourism']
        index = [np.array(['Precision']*8 + ['Recall']*8 + ['Fscore']*8), np.array(columns*3)]
        models_dict = {}
        for model_name in model_report:

            report = model_report[model_name]
            precision = report['precision']
            recall = report['recall']
            fscore = report['fscore']
            print("preree")
            print(precision)
            precision_means = {}
            recall_means = {}
            fscore_means = {}
            for column in columns:
                precision_means[column] = st.mean(precision[column].tolist())
                recall_means[column] = st.mean(recall[column].tolist())
                fscore_means[column] = st.mean(fscore[column].tolist())

            model_metrics = []

            for column in columns:
                model_metrics.append(precision_means[column])
            for column in columns:
                model_metrics.append(recall_means[column])
            for column in columns:
                model_metrics.append(fscore_means[column])

            models_dict[model_name] = model_metrics

        print("dddd")
        print(len(models_dict['mfa']))
        print(len(models_dict['stf']))
        df = pd.DataFrame(models_dict, index=index)

        print(df)

        output_dir = "output/performance_plots/" + dataset_type_dir + category_type_dir
        output = output_dir + str(n_splits) + "_folds/" + str(n_replications) + "_replications/"
        Path(output).mkdir(parents=True, exist_ok=True)
        writer = pd.ExcelWriter(output + 'metrics.xlsx', engine='xlsxwriter')

        df.to_excel(writer, sheet_name='Sheet1')

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()