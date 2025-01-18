import pandas as pd
from typing import Any, List, Dict
import numpy as np

import sklearn
from sklearn import svm
from sklearn import ensemble
from sklearn import covariance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import pickle
import dcor

import warnings

warnings.filterwarnings("ignore")


def perf_measure(y_actual: List[Any], y_predicted: List[Any]) -> Dict:
    ans = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
           'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'TPR': 0, 'FPR': 0,
           'F': 0, 'MCC': 0}
    for i in range(len(y_predicted)):
        if y_actual[i] == 0 and y_predicted[i] == 1:
            ans['TP'] += 1
        if y_predicted[i] == 1 and y_actual[i] != 0:
            ans['FP'] += 1
        if y_actual[i] != 0 and y_predicted[i] == -1:
            ans['TN'] += 1
        if y_predicted[i] == -1 and y_actual[i] == 0:
            ans['FN'] += 1
    try:
        ans['Precision'] = ans['TP'] / (ans['TP'] + ans['FP'])
    except ZeroDivisionError:
        ans['Precision'] = 'not defined'
    try:
        ans['Recall'] = ans['TP'] / (ans['TP'] + ans['FN'])
    except ZeroDivisionError:
        ans['Recall'] = 'not defined'
    try:
        ans['Accuracy'] = (ans['TP'] + ans['TN']) / (ans['FP'] + ans['FN'] + ans['TP'] + ans['TN'])
    except ZeroDivisionError:
        ans['Accuracy'] = 'not defined'
    try:
        ans['F'] = 2 * ans['Precision'] * ans['Recall'] / (ans['Precision'] + ans['Recall'])
    except ZeroDivisionError:
        ans['F'] = 'not defined'
    except BaseException:
        ans['F'] = 'can not be computed'
    return ans


def get_metrics(clf, X_train, X_test, X_test_full, y_train, y_test, y_test_full, metrics_filename, model_name,
                dict_letter, dataset):
    pickle.dump(clf, open(Path('models') / model_name, "wb"))
    with open(Path('results/anomaly') / metrics_filename, 'w') as f:
        f.write('train\n')
        predicted = clf.predict(X_train)
        f.write(f"{perf_measure(y_actual=y_train.to_list(), y_predicted=list(predicted))}\n\n")
        f.write('test\n')
        predicted = clf.predict(X_test)
        f.write(f"{perf_measure(y_actual=y_test.to_list(), y_predicted=list(predicted))}\n\n")
        f.write('full_test\n')
        predicted = clf.predict(X_test_full)
        labels = []
        for y in y_test_full:
            if y == 0:
                labels.append(1)
            else:
                labels.append(-1)
        cm = confusion_matrix(labels, predicted)

        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f"{dict_letter}")

        if 'forest' in model_name:
            xxx = 'forest'
        elif 'elliptic' in model_name:
            xxx = 'elliptic'
        elif 'SVM' in model_name:
            xxx = 'SVM'
        else:
            raise BaseException('UNEXPECTED MODEL')
        cm_filename = f"{dataset}_anomaly_cm_{xxx}.png"
        plt.savefig(Path("conf_matrix") / cm_filename, bbox_inches='tight')
        plt.close()
        f.write(f"{perf_measure(y_actual=y_test_full.to_list(), y_predicted=list(predicted))}\n\n")


input_data_path = [
    # 'tabular_data/CIFAR10/merged_data_all_jsma.csv',
    'tabular_data/CIFAR10GRAY/merged_data_all_jsma_cured.csv',
    'tabular_data/MNIST/merged_data_all_jsma.csv',
    'tabular_data/LaVAN/merged_data.csv'
]

for data_path in input_data_path:
    dataset = data_path.split('/')[1]
    dict_letter = {'CIFAR10': 'a', 'CIFAR10GRAY': 'b', 'MNIST': 'c', 'LaVAN': 'd'}
    d = pd.read_csv(data_path, index_col=0, encoding="windows-1251")
    d = d.dropna(axis=0)
    # d = d.loc[d['flag'] == 0]
    # data = d.drop(['title'], axis=1)
    #
    # corr = data.corr(method='kendall')  # todo сохранить картинку
    # # corr = data.corr(method=dcor.distance_correlation)
    # corr = corr.replace(np.nan, 0)
    #
    # plt.figure(figsize=(20, 20))
    # cmap = sns.diverging_palette(240, 240, as_cmap=True)
    # sns.heatmap(corr, annot=True, vmin=-1, vmax=1, fmt=".2g", cmap=cmap)
    # corr_filename = f"{dataset}_kendall_corr.png"
    # plt.title(f"{dataset}")
    # plt.savefig(Path("corr_matrix") / corr_filename, bbox_inches='tight')
    # plt.close()

    data = d.loc[d['flag'] == 0]
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=3802)

    dropped = ['flag', 'title']

    if dataset == 'CIFAR10GRAY':  # 54
        dropped.extend(['mean', 'std', '75q', '95q', '97q', '99q', 'iqr', '3iqr_cnt', '6iqr_cnt', 'skew'])
    elif dataset == 'CIFAR10':  # 9
        dropped.extend(['median', 'var', '25q', '97q', 'iqr', '3sigma_cnt', '3iqr_cnt', '6iqr_cnt', 'kurt'])
    elif dataset == 'MNIST':  # 23
        dropped.extend(['75q', '95q', '5sigma_cnt', '3iqr_cnt', '6iqr_cnt'])
    elif dataset == 'LaVAN':
        dropped.extend(['iqr', 'var', 'max', 'skew'])

    X_train, y_train = train_data.drop(dropped, axis=1), train_data['flag']
    X_test, y_test = test_data.drop(dropped, axis=1), test_data['flag']
    data = pd.read_csv(data_path, index_col=0, encoding="windows-1251")
    data = data.dropna(axis=0)
    X_test_full, y_test_full = data.drop(dropped, axis=1), data['flag']

    top_F = 0
    top_model = ''
    kernels_one_class = ['linear', 'rbf', 'poly']
    models = ['forest', 'elliptic', 'SVM']
    for model in models:
        if model == 'forest':
            best_F = 0
            for n_estimators in range(23, 64):
                print(f"n_estimators:\t{n_estimators}")
                model_name = f'{model}_{dataset}.pkl'
                metrics_filename = f"{data_path[data_path.index('/') + 1:data_path.rindex('/')]}_{model}_{n_estimators}.txt"
                clf = ensemble.IsolationForest(n_estimators=n_estimators, random_state=3802)
                clf.fit(X_train)
                predicted = clf.predict(X_test_full)
                F = perf_measure(y_actual=y_test_full.to_list(), y_predicted=list(predicted))['F']
                if F > best_F:
                    best_F = F
                    get_metrics(clf, X_train, X_test, X_test_full,
                                y_train, y_test, y_test_full,
                                metrics_filename, model_name, dict_letter[dataset], dataset)
                    if F > top_F:
                        top_F = F
                        top_model = model_name
            print(dropped)
            print(best_F)
        elif model == 'SVM':
            best_F = 0
            for kernel in kernels_one_class:
                nu = 0
                while nu < 0.5:
                    print(f"nu:\t{nu}")
                    nu += 0.01
                    model_name = f'{model}_{dataset}.pkl'
                    metrics_filename = f"{data_path[data_path.index('/') + 1:data_path.rindex('/')]}_{model}_{kernel}_{nu}.txt"
                    if kernel == 'poly':
                        for deg in range(1, 3):
                            clf = sklearn.svm.OneClassSVM(nu=nu, kernel=kernel, degree=deg)
                            clf.fit(X_train)
                            predicted = clf.predict(X_test_full)
                            F = perf_measure(y_actual=y_test_full.to_list(), y_predicted=list(predicted))['F']
                            if F > best_F:
                                best_F = F
                                get_metrics(clf, X_train, X_test, X_test_full,
                                            y_train, y_test, y_test_full,
                                            metrics_filename, model_name, dict_letter[dataset], dataset)
                                if F > top_F:
                                    top_F = F
                                    top_model = model_name
                    else:
                        clf = sklearn.svm.OneClassSVM(nu=nu, kernel=kernel)
                        clf.fit(X_train)
                        predicted = clf.predict(X_test_full)
                        F = perf_measure(y_actual=y_test_full.to_list(), y_predicted=list(predicted))['F']
                        if F > best_F:
                            best_F = F
                            get_metrics(clf, X_train, X_test, X_test_full,
                                        y_train, y_test, y_test_full,
                                        metrics_filename, model_name, dict_letter[dataset], dataset)
                            if F > top_F:
                                top_F = F
                                top_model = model_name
        elif model == 'elliptic':
            best_F = 0
            model_name = f'{model}_{dataset}.pkl'
            metrics_filename = f"{data_path[data_path.index('/') + 1:data_path.rindex('/')]}_{model}.txt"
            clf = covariance.EllipticEnvelope(random_state=3802)
            clf.fit(X_train)
            predicted = clf.predict(X_test_full)
            F = perf_measure(y_actual=y_test_full.to_list(), y_predicted=list(predicted))['F']
            if F > best_F:
                best_F = F
                get_metrics(clf, X_train, X_test, X_test_full,
                            y_train, y_test, y_test_full,
                            metrics_filename, model_name, dict_letter[dataset], dataset)
                if F > top_F:
                    top_F = F
                    top_model = model_name
        else:
            raise BaseException("unexpected model")

    # print(top_F)
    # print(top_model)
