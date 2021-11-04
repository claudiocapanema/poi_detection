from pathlib import Path
from matplotlib import pyplot
import numpy as np
import pandas as pd

class NextPoiCategoryPredictionLoader:

    def __init__(self):
        pass

    def plot_history_metrics(self, folds_histories, folds_reports, output_dir, n_folds, n_replications, show=False):

        # n_folds = len(folds_histories)
        # n_replications = len(folds_histories[0])
        output_dir = output_dir + str(n_folds) + "_folds/" + str(n_replications) + "_replications/"
        print("pasta: ", output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for fold_histories in folds_histories:
            for i in range(len(fold_histories)):
                h = fold_histories[i]
                file_index = "replication_" + str(i)
                pyplot.figure(figsize=(12, 12))
                pyplot.plot(h['acc'])
                pyplot.plot(h['val_acc'])
                pyplot.title('model acc')
                pyplot.ylabel('acc')
                pyplot.xlabel('epoch')
                pyplot.legend(['train', 'test'], loc='upper left')
                if show:
                    pyplot.show()
                pyplot.savefig(output_dir + file_index+ "_history_accuracy.png")
                # summarize history for loss
                pyplot.figure(figsize=(12, 12))
                pyplot.plot(h['loss'])
                pyplot.plot(h['val_loss'])
                pyplot.title('model loss')
                pyplot.ylabel('loss')
                pyplot.xlabel('epoch')
                pyplot.legend(['train', 'test'], loc='upper left')
                pyplot.savefig(output_dir + file_index + "_history_loss.png")
                if show:
                    pyplot.show()

    def save_report_to_csv(self, output_dir, report, n_folds, n_replications, usuarios):

        precision_dict = {}
        recall_dict = {}
        fscore_dict = {}
        column_size = n_folds * n_replications
        print("final")
        print(report)
        for key in report:
            if key == 'accuracy':
                fscore_column = 'accuracy'
                fscore_dict[fscore_column] = report[key]
                continue
            elif key == 'recall' or key == 'f1-score' \
                    or key == 'support':
                continue
            elif key == 'macro avg' or key == 'weighted avg':
                precision_dict[key] = report[key]['precision']
                recall_dict[key] = report[key]['recall']
                fscore_dict[key] = report[key]['f1-score']
                continue
            fscore_column = key
            fscore_column_data = report[key]['f1-score']
            if len(fscore_column_data) < column_size:
                while len(fscore_column_data) < column_size:
                    fscore_column_data.append(np.nan)
            fscore_dict[fscore_column] = fscore_column_data

            precision_column = key
            precision_column_data = report[key]['precision']
            if len(precision_column_data) < column_size:
                while len(precision_column_data) < column_size:
                    precision_column_data.append(np.nan)
            precision_dict[precision_column] = precision_column_data

            recall_column = key
            recall_column_data = report[key]['recall']
            if len(recall_column_data) < column_size:
                while len(recall_column_data) < column_size:
                    recall_column_data.append(np.nan)
            recall_dict[recall_column] = recall_column_data

        #print("final: ", new_dict)
        precision = pd.DataFrame(precision_dict)
        print("Métricas precision: \n", precision)
        output_dir = output_dir + str(n_folds) + "_folds/" + str(n_replications) + "_replications/"
        print("pasta", output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        precision.to_csv(output_dir + "precision.csv", index_label=False, index=False)

        recall = pd.DataFrame(recall_dict)
        print("Métricas recall: \n", recall)
        recall.to_csv(output_dir + "recall.csv", index_label=False, index=False)

        fscore = pd.DataFrame(fscore_dict)
        print("Métricas fscore: \n", fscore)
        fscore.to_csv(output_dir + "fscore.csv", index_label=False, index=False)

    def wrong_predicted_samples_plots(self, output_dir, x, y_predicted, y_label):
        pass
