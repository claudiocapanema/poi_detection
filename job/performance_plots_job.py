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
from loader.performance_plots_loader import PerformancePlotsLoader
from foundation.configuration.input import Input
from extractor.file_extractor import FileExtractor

class PerformancePlots(Job):

    def __init__(self):
        self.user_step_domain = UserStepDomain()
        self.points_of_interest_domain = PointsOfInterestDomain()
        self.next_poi_category_prediction_configuration = NextPoiCategoryPredictionConfiguration()
        self.next_poi_category_prediction_domain = NextPoiCategoryPredictionDomain(
            Input.get_instance().inputs['dataset_name'], 0, 0)
        self.file_loader = FileLoader()
        self.performance_plots_loader = PerformancePlotsLoader(Input.get_instance().inputs['dataset_name'])
        self.file_extractor = FileExtractor()

    def start(self):
        dataset_name = Input.get_instance().inputs['dataset_name']
        categories_type = Input.get_instance().inputs['categories_type']

        n_splits = self.next_poi_category_prediction_configuration.N_SPLITS[1]
        n_replications = self.next_poi_category_prediction_configuration.N_REPLICATIONS[1]
        output_base_dir = self.next_poi_category_prediction_configuration.OUTPUT_BASE_DIR[1]
        dataset_type_dir = self.next_poi_category_prediction_configuration.DATASET_TYPE[1][dataset_name]
        category_type_dir = self.next_poi_category_prediction_configuration.CATEGORY_TYPE[1][categories_type]

        self.performance_plots_loader.export_reports(n_splits, n_replications, output_base_dir, dataset_type_dir, category_type_dir, dataset_name)