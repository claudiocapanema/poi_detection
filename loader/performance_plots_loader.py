import seaborn as sns

import statistics as st
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from configuration.next_poi_category_prediciton_configuration import NextPoiCategoryPredictionConfiguration
from domain.next_poi_category_prediction_domain import NextPoiCategoryPredictionDomain
from extractor.file_extractor import FileExtractor
from foundation.util.statistics_utils import t_distribution_test

class PerformancePlotsLoader:

    def __init__(self, dataseet_name):
        self.next_poi_category_prediction_configuration = NextPoiCategoryPredictionConfiguration()
        self.next_poi_category_prediction_domain = NextPoiCategoryPredictionDomain(dataseet_name, 0, 0)
        self.file_extractor = FileExtractor()
        self.columns = {'users_steps': ['Home', 'Work', 'Other', 'Shopping', 'Community', 'Food', 'Entertainment', 'Travel', 'Outdoors', 'Nightlife'],
                        'gowalla': ['Shopping', 'Community', 'Food', 'Entertainment', 'Travel', 'Outdoors', 'Nightlife']}
        self.y_limits = {'users_steps': {'Accuracy': 0.68, 'Macro f1-score': 0.33, 'Weighted f1-score': 0.6},
                         'gowalla': {'Accuracy': 0.48, 'Macro f1-score': 0.4, 'Weighted f1-score': 0.46}}

    def _convert_names(self, names):

        convert_dict = {'mfa': 'POI-RGNN', 'stf': 'STF-RNN', 'map': 'MAP', 'serm': 'SERM', 'next': 'MHA+PE', 'garg': 'GARG'}

        for i in range(len(names)):

            names[i] = convert_dict[names[i]]

        return names

    def plot_general_metrics(self, report, columns, base_dir, dataset):

        sns.set_theme('whitegrid')
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
        self.barplot_with_values(metrics, 'Solution', 'Accuracy', base_dir, filename, title, ax, 0)

        #title = 'Macro average fscore'
        filename = 'barplot_macro_avg_fscore'
        self.barplot_with_values(metrics, 'Solution', 'Macro f1-score', base_dir, filename, title, ax, 1)

        #title = 'Weighted average fscore'
        filename = 'barplot_weighted_avg_fscore'
        self.barplot_with_values(metrics, 'Solution', 'Weighted f1-score', base_dir, filename, title, ax, 2)

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

    def plot_general_metrics_with_confidential_interval(self, report, columns, base_dir, dataset):

        sns.set_theme(style='whitegrid')
        macro_fscore_list = []
        model_name_list = []
        accuracy_list = []
        weighted_fscore_list = []
        for model_name in report:

            fscore = report[model_name]['fscore']
            accuracy = np.round(fscore['accuracy'].to_numpy() * 100, 1).tolist()
            weighted_fscore = np.round(fscore['weighted avg'].to_numpy() * 100, 1).tolist()
            macro_fscore = np.round(fscore['macro avg'].to_numpy() * 100, 1).tolist()

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
        # mpl.use("pgf")
        # mpl.rcParams.update({
        #     "pgf.texsystem": "pdflatex",
        #     'font.family': 'serif',
        #     'text.usetex': True,
        #     'pgf.rcfonts': False,
        # })
        sns.set(style='whitegrid')

        #fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, figsize=(14,20), tight_layout=True)
        fig, ax = plt.subplots(ncols=3, nrows=1, sharey=True, sharex=True, figsize=(35, 15), tight_layout=True)



        self.barplot_with_values(metrics, 'Solution', 'Accuracy', base_dir, filename, title, dataset, ax, 2)

        #title = 'Macro average fscore'
        filename = 'barplot_macro_avg_fscore_ci'
        self.barplot_with_values(metrics, 'Solution', 'Macro f1-score', base_dir, filename, title, dataset, ax, 0)

        #title = 'Weighted average fscore'
        filename = 'barplot_weighted_avg_fscore_ci'
        self.barplot_with_values(metrics, 'Solution', 'Weighted f1-score', base_dir, filename, title, dataset, ax, 1)
        #plt.ylim(0, 0.5)
        #plt.tick_params(labelsize=18)
        #plt.grid(True)
        #fig.subplots_adjust(top=0.5, bottom=0, left=0, right=1)

        #fig.tight_layout(pad=1)
        #plt.savefig(dataset + '_metrics_horizontal_latex.pgf')
        fig.savefig(base_dir + dataset + "_metrics_horizontal.png", bbox_inches='tight', dpi=400)
        fig.savefig(base_dir + dataset + "_metrics_horizontal.svg", bbox_inches='tight', dpi=400)
        plt.figure()

        metrics_stacked = metrics[['Solution', 'Accuracy']]
        metrics_stacked.columns = ['Solution', 'Performance']
        metrics_stacked['Metric'] = np.array(['Accuracy'] * len(metrics_stacked))

        macro_df = metrics[['Solution', 'Macro f1-score']]
        macro = np.array(macro_df['Macro f1-score'].tolist())
        macro_df.columns = ['Solution', 'Performance']
        macro_df['Metric'] = np.array(['Macro f1-score'] * len(macro))

        weighted_df = metrics[['Solution', 'Weighted f1-score']]
        weighted = np.array(weighted_df['Weighted f1-score'].tolist())
        weighted_df.columns = ['Solution', 'Performance']
        weighted_df['Metric'] = np.array(['Weighted f1-score'] * len(weighted))

        metrics_stacked = pd.DataFrame(metrics_stacked.to_dict())
        macro_df = pd.DataFrame(macro_df.to_dict())
        weighted_df = pd.DataFrame(weighted_df.to_dict())

        solutions = metrics_stacked['Solution'].tolist() + macro_df['Solution'].tolist() + weighted_df['Solution'].tolist()
        performance = metrics_stacked['Performance'].tolist() + macro_df['Performance'].tolist() + weighted_df['Performance'].tolist()
        metric = metrics_stacked['Metric'].tolist() + macro_df['Metric'].tolist() + weighted_df['Metric'].tolist()

        # plt.figure()
        # sns.set(font_scale=1.2, style='whitegrid')
        # metrics_stacked_new = pd.DataFrame({'Solution': solutions, 'Performance': performance, 'Metric': metric})
        # print("concatenar: ", metrics_stacked_new)
        # g = sns.FacetGrid(metrics_stacked, row="Metric")
        # g.map(sns.barplot, 'Solution', 'Performance')
        # g.savefig(dataset + "_metrics.png")

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

    def barplot_with_values(self, metrics, x_column, y_column, base_dir, file_name, title, dataset, ax, index):
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        #plt.figure(figsize=(8, 5))
        #sns.set(font_scale=1.2, style='whitegrid')
        # if y_column == 'Macro f1-score':
        #     order = ['MAP', 'STF-RNN', 'MHSA+PE', 'SERM', 'GARG', 'MFA-RNN']
        # elif y_column == 'Accuracy':
        #     order = ['STF-RNN', 'MAP', 'MHSA+PE', 'SERM', 'GARG', 'MFA-RNN']
        # else:
        #     order = ['MAP', 'STF-RNN', 'MHSA+PE', 'SERM', 'GARG', 'MFA-RNN']
        # order = list(reversed(order))
        #ax[index].set_ylim(0, 0.5)
        size = 35
        sorted_values = sorted(metrics[y_column].tolist())
        maximum = sorted_values[-1]
        if dataset == "users_steps":
            #ax[index].set_ylim(0, maximum * 1.14)
            y_labels = {'macro': [40, 46, 40, 45, 45, 55], 'weighted': [32, 32, 32, 32, 32, 32],
                        'accuracy': [32, 32, 32, 32, 32, 32]}
            #y_labels = {'macro': [22, 22, 22, 22, 22, 22], 'weighted': [22, 22, 22, 22, 22, 22], 'accuracy': [22, 22, 22, 22, 22, 22]}
        else:
            y_labels = {'macro': [30, 30, 30, 30, 30, 30], 'weighted': [30, 30, 30, 30, 30, 30],
                        'accuracy': [30, 30, 30, 30, 30, 30]}
            #ax[index].set_ylim(0, maximum * 1.14)
            if 'weighted' in file_name:
                #ax[index].set_ylim(0, maximum * 1.2)
                pass
        plt.ylim(0, maximum*1.2)

        #ax[index].set_aspect(5)
        figure = sns.barplot(x=x_column, y=y_column, data=metrics, ax=ax[index])

        figure.set_ylabel(y_column, fontsize=size)
        figure.set_xlabel(x_column, fontsize=size)
        # figure0.tick_params(labelsize=10)
        y_label = "accuracy"
        count = 0


        if "macro" in file_name:
            y_label = "macro"
        elif "weighted" in file_name:
            y_label = "weighted"
        for p in figure.patches:
            figure.annotate(format(p.get_height(), '.1f'),
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center',
                            size=size,
                             xytext=(0, y_labels[y_label][count]),
                             textcoords='offset points')
            count += 1
        ax[index].tick_params(axis='x', labelsize=size - 4, rotation=40)
        ax[index].tick_params(axis='y', labelsize=size - 4)
        figure = figure.get_figure()
        # plt.legend(bbox_to_anchor=(0.65, 0.74),
        #            borderaxespad=0)

        # ax.yticks(labels=[df['Precision'].tolist()])
        #figure.savefig(base_dir + file_name + ".png", bbox_inches='tight', dpi=400)
        #plt.figure()

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
            precision = report['precision']*100
            recall = report['recall']*100
            fscore = report['fscore']*100
            precision_means = {}
            recall_means = {}
            fscore_means = {}
            for column in columns:
                precision_means[column] = t_distribution_test(precision[column].tolist())
                recall_means[column] = t_distribution_test(recall[column].tolist())
                fscore_means[column] = t_distribution_test(fscore[column].tolist())

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
        df = pd.DataFrame(models_dict, index=index).round(4)

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

        max_values = self.idmax(df)
        print("zzz", max_values)
        max_columns = {'mfa': [], 'stf': [], 'map': [], 'serm': [], 'next': [], 'garg': []}

        for max_value in max_values:
            row_index = max_value[0]
            column = max_value[1]
            column_values = df[column].tolist()
            column_values[row_index] = "textbf{" + str(column_values[row_index]) + "}"

            df[column] = np.array(column_values)

        df.columns = ['POI-RGNN', 'STF-RNN', 'MAP', 'SERM', 'MHA+PE', 'GARG']

        # get improvements
        poi_rgnn = [float(i.replace("textbf{", "").replace("}", "")[:4]) for i in df['POI-RGNN'].to_numpy()]
        stf = [float(i.replace("textbf{", "").replace("}", "")[:4]) for i in df['STF-RNN'].to_numpy()]
        map = [float(i.replace("textbf{", "").replace("}", "")[:4]) for i in df['MAP'].to_numpy()]
        serm = [float(i.replace("textbf{", "").replace("}", "")[:4]) for i in df['SERM'].to_numpy()]
        mha = [float(i.replace("textbf{", "").replace("}", "")[:4]) for i in df['MHA+PE'].to_numpy()]
        garg = [float(i.replace("textbf{", "").replace("}", "")[:4]) for i in df['GARG'].to_numpy()]
        difference = []

        init = {'gowalla': 14, 'users_steps': 20}
        for i in range(init[dataset_name], len(poi_rgnn)):
            min_ = max([stf[i], map[i], serm[i], mha[i], garg[i]])
            max_ = min([stf[i], map[i], serm[i], mha[i], garg[i]])
            value = poi_rgnn[i]
            if min_ < value:
                min_ = value - min_
            else:
                min_ = 0
            if max_ < value:
                max_ = value - max_
            else:
                max_ = 0

            s = str(round(min_, 1)) + "\%--" + str(round(max_, 1)) + "\%"
            difference.append(
                [round(value, 1), round(stf[i], 1), round(map[i], 1), round(serm[i], 1), round(mha[i], 1), round(garg[i], 1), round(min_, 1), round(max_, 1), s])

        difference_df = pd.DataFrame(difference, columns=['base', 'stf', 'map', 'serm', 'mha', 'garg', 'min', 'max', 'texto'])

        difference_df.to_csv(output + "difference.csv", index=False)


        latex = df.to_latex().replace("\}", "}").replace("\{", "{").replace("\\\nRecall", "\\\n\hline\nRecall").replace("\\\nF-score", "\\\n\hline\nF1-score")
        pd.DataFrame({'latex': [latex]}).to_csv(output + "latex.txt", header=False, index=False)

        self.plot_general_metrics_with_confidential_interval(model_report, columns, output+dataset_name, dataset_name)

    def idmax(self, df):

        df_indexes = []
        columns = df.columns.tolist()
        print("colunas", columns)
        for i in range(len(df)):

            row = df.iloc[i].tolist()
            print("ddd", row)
            indexes = self.select_mean(i, row, columns)
            df_indexes += indexes

        return df_indexes

    def select_mean(self, index, values, columns):

        list_of_means = []
        indexes = []

        for i in range(len(values)):

            value = float(values[i][:4])
            list_of_means.append(value)

        max_value = max(list_of_means)

        for i in range(len(list_of_means)):

            if list_of_means[i] == max_value:
                indexes.append([index, columns[i]])

        return indexes