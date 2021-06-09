import seaborn as sns

import statistics as st
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from configuration.next_poi_category_prediciton_configuration import NextPoiCategoryPredictionConfiguration
from domain.next_poi_category_prediction_domain import NextPoiCategoryPredictionDomain
from extractor.file_extractor import FileExtractor

class PerformancePlotsLoader:

    def __init__(self, dataseet_name):
        self.next_poi_category_prediction_configuration = NextPoiCategoryPredictionConfiguration()
        self.next_poi_category_prediction_domain = NextPoiCategoryPredictionDomain(dataseet_name, 0, 0)
        self.file_extractor = FileExtractor()
        self.columns = {'users_steps': ['Home', 'Work', 'Other', 'Commuting', 'Amenity', 'Leisure', 'Shop', 'Tourism'],
                        'gowalla': ['Shopping', 'Community', 'Food', 'Entertainment', 'Travel', 'Outdoors', 'Nightlife']}

    def plot_general_metrics(self, report, columns, base_dir):

        sns.set_theme()
        macro_fscore_list = []
        model_name_list = []
        accuracy_list = []
        weighted_fscore_list = []
        for model_name in report:

            fscore = report[model_name]['fscore']
            accuracy = st.mean(fscore['accuracy'].tolist())
            weighted_fscore = st.mean(fscore['weighted avg'].tolist())
            total_fscore = 0
            for column in columns:
                total_fscore += st.mean(fscore[column].tolist())

            macro_fscore = total_fscore/len(columns)
            macro_fscore_list.append(macro_fscore)
            model_name_list.append(self.next_poi_category_prediction_configuration.FORMAT_MODEL_NAME[1][model_name])
            accuracy_list.append(accuracy)
            weighted_fscore_list.append(weighted_fscore)

        metrics = pd.DataFrame({'Solution': model_name_list, 'Accuracy': accuracy_list,
                                'Macro f1-score': macro_fscore_list, 'Weighted f1-score': weighted_fscore_list})

        print(metrics)
        title = ''
        filename = 'barplot_accuracy'
        self.barplot_with_values(metrics, 'Solution', 'Accuracy', base_dir, filename, title)

        #title = 'Macro average fscore'
        filename = 'barplot_macro_avg_fscore'
        self.barplot_with_values(metrics, 'Solution', 'Macro f1-score', base_dir, filename, title)

        #title = 'Weighted average fscore'
        filename = 'barplot_weighted_avg_fscore'
        self.barplot_with_values(metrics, 'Solution', 'Weighted f1-score', base_dir, filename, title)

        # columns = list(metrics.columns)
        # print("antigas: ", columns)
        # columns = list(osm_categories_to_int.sub_category()) + [columns[-4], columns[-3], columns[-2], columns[-1]]
        # columns = [e.replace("/","_") for e in columns]
        # metrics.columns = columns
        # metrics.to_csv("metricas_totais.csv", index=False, index_label=False)
        # print("novas colunas\n", metrics)
        # for i in range(len(list(osm_categories_to_int.sub_category()))):
        #     title = 'F-score'
        #     filename = folds_replications_filename + '_barplot_' + columns[i] + "_fscore"
        #     self.barplot(metrics, 'Method', columns[i], base_dir, filename, title)

    def plot_general_metrics_with_confidential_interval(self, report, columns, base_dir):

        sns.set_theme()
        macro_fscore_list = []
        model_name_list = []
        accuracy_list = []
        weighted_fscore_list = []
        for model_name in report:

            fscore = report[model_name]['fscore']
            accuracy = fscore['accuracy'].tolist()
            weighted_fscore = fscore['weighted avg'].tolist()
            macro_fscore = fscore['macro avg'].tolist()

            # total_fscore = 0
            # for column in columns:
            #     total_fscore += fscore[column].tolist()
            #
            # macro_fscore = total_fscore/len(columns)
            macro_fscore_list += macro_fscore
            model_name_list += [self.next_poi_category_prediction_configuration.FORMAT_MODEL_NAME[1][model_name]]*len(accuracy)
            accuracy_list += accuracy
            weighted_fscore_list += weighted_fscore

        metrics = pd.DataFrame({'Solution': model_name_list, 'Accuracy': accuracy_list,
                                'Macro f1-score': macro_fscore_list, 'Weighted f1-score': weighted_fscore_list})

        print(metrics)
        title = ''
        filename = 'barplot_accuracy_ci'
        self.barplot_with_values(metrics, 'Solution', 'Accuracy', base_dir, filename, title)

        #title = 'Macro average fscore'
        filename = 'barplot_macro_avg_fscore_ci'
        self.barplot_with_values(metrics, 'Solution', 'Macro f1-score', base_dir, filename, title)

        #title = 'Weighted average fscore'
        filename = 'barplot_weighted_avg_fscore_ci'
        self.barplot_with_values(metrics, 'Solution', 'Weighted f1-score', base_dir, filename, title)

        # columns = list(metrics.columns)
        # print("antigas: ", columns)
        # columns = list(osm_categories_to_int.sub_category()) + [columns[-4], columns[-3], columns[-2], columns[-1]]
        # columns = [e.replace("/","_") for e in columns]
        # metrics.columns = columns
        # metrics.to_csv("metricas_totais.csv", index=False, index_label=False)
        # print("novas colunas\n", metrics)
        # for i in range(len(list(osm_categories_to_int.sub_category()))):
        #     title = 'F-score'
        #     filename = folds_replications_filename + '_barplot_' + columns[i] + "_fscore"
        #     self.barplot(metrics, 'Method', columns[i], base_dir, filename, title)

    def barplot(self, metrics, x_column, y_column, base_dir, file_name, title):
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        plt.figure()
        figure = sns.barplot(x=x_column, y=y_column, data=metrics).set_title(title)
        figure = figure.get_figure()
        figure.savefig(base_dir + file_name + ".png", bbox_inches='tight', dpi=400)

    def barplot_with_values(self, metrics, x_column, y_column, base_dir, file_name, title):
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        plt.figure()
        figure = sns.barplot(x=x_column, y=y_column, data=metrics)

        # figure.set_ylabel(x_column, fontsize=15)
        # figure.set_xlabel(y_column, fontsize=15)
        # figure0.tick_params(labelsize=10)
        y_label = "accuracy"
        count = 0
        y_labels = {'macro': [17, 24, 20, 17, 20, 20], 'weighted': [11, 11, 11, 11, 11, 11], 'accuracy': [11, 11, 11, 11, 11, 11]}
        if "macro" in file_name:
            y_label = "macro"
        elif "weighted" in file_name:
            y_label = "weighted"
        for p in figure.patches:
            figure.annotate(format(p.get_height(), '.2f'),
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center',
                             xytext=(0, y_labels[y_label][count]),
                             textcoords='offset points')
            count += 1
        figure = figure.get_figure()
        # plt.legend(bbox_to_anchor=(0.65, 0.74),
        #            borderaxespad=0)
        maximum_value = metrics[y_column].max()
        plt.ylim(0, maximum_value + 0.1)
        # ax.yticks(labels=[df['Precision'].tolist()])
        figure.savefig(base_dir + file_name + ".png", bbox_inches='tight', dpi=400)
        plt.figure()

    def export_reports(self, n_splits, n_replications, output_base_dir, dataset_type_dir, category_type_dir, dataset_name):

        model_report = {'mfa': {}, 'stf': {}, 'map': {}, 'serm': {}, 'next': {}, 'garg': {}}
        for model_name in model_report.keys():
            model_name_dir = self.next_poi_category_prediction_configuration.MODEL_NAME[1][model_name]
            output_dir = self.next_poi_category_prediction_domain. \
                output_dir(output_base_dir, dataset_type_dir, category_type_dir, model_name_dir)
            output = output_dir + str(n_splits) + "_folds/" + str(n_replications) + "_replications/"

            model_report[model_name]['precision'] = self.file_extractor.read_csv(output + "precision.csv").round(4)
            model_report[model_name]['recall'] = self.file_extractor.read_csv(output + "recall.csv").round(4)
            model_report[model_name]['fscore'] = self.file_extractor.read_csv(output + "fscore.csv").round(4)

        print(model_report)
        columns = self.columns[dataset_name]
        index = [np.array(['Precision'] * len(columns) + ['Recall'] * len(columns) + ['Fscore'] * len(columns)), np.array(columns * 3)]
        models_dict = {}
        for model_name in model_report:

            report = model_report[model_name]
            precision = report['precision']
            recall = report['recall']
            fscore = report['fscore']
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
        print(len(models_dict['map']))
        print(len(models_dict['serm']))
        print(len(models_dict['garg']))
        print(len(models_dict['next']))
        print(len(models_dict['stf']))
        df = pd.DataFrame(models_dict, index=index).round(2)

        print(df)

        output_dir = "output/performance_plots/" + dataset_type_dir + category_type_dir
        output = output_dir + str(n_splits) + "_folds/" + str(n_replications) + "_replications/"
        Path(output).mkdir(parents=True, exist_ok=True)
        # writer = pd.ExcelWriter(output + 'metrics.xlsx', engine='xlsxwriter')
        #
        # df.to_excel(writer, sheet_name='Sheet1')
        #
        # # Close the Pandas Excel writer and output the Excel file.
        # writer.save()

        max_values = df.idxmax(axis=1)
        max_values = max_values.tolist()
        print("zzz", max_values)
        max_columns = {'mfa': [], 'stf': [], 'map': [], 'serm': [], 'next': [], 'garg': []}
        for i in range(len(max_values)):
            e = max_values[i]
            max_columns[e].append(i)

        for key in max_columns:
            column_values = df[key].tolist()

            column_list = max_columns[key]
            for j in range(len(column_list)):
                k = column_list[j]
                column_values[k] = "textbf{" + str(column_values[k]) + "}"

            df[key] = np.array(column_values)
        # for i in range(len(df)):
        #     column = max_values[i]
        #     df.at[i,column] = "'\textbf{" + str(df.iloc[i][column]) + "}"

        latex = df.to_latex().replace("\}", "}").replace("\{", "{").replace("\\\nRecall", "\\\n\hline\nRecall").replace("\\\nF-score", "\\\n\hline\nF-score")
        pd.DataFrame({'latex': [latex]}).to_csv(output + "latex.txt", header=False, index=False)

        self.plot_general_metrics_with_confidential_interval(model_report, columns, output+dataset_name)

