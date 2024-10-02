import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, \
    recall_score

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import pickle
import warnings

warnings.filterwarnings("ignore")


def save_metrics(y_test, preds, binary=False):
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'f1': f1_score(y_test, preds, average="binary" if binary else "macro"),
        'precision': precision_score(y_test, preds, average="binary" if binary else None),
        'recall': recall_score(y_test, preds, average="binary" if binary else None),
    }

    return metrics


def get_metrics(clf, X_train, X_test, X_test_full, y_train, y_test, y_test_full, metrics_filename, model_name, dict_letter, dataset):
    pickle.dump(clf, open(Path('models') / model_name, "wb"))
    with open(Path('classifications') / metrics_filename, 'w') as f:
        f.write('train\n')
        predicted = clf.predict(X_train)
        f.write(f"{save_metrics(y_train.to_list(), list(predicted))}\n\n")
        f.write('test\n')
        predicted = clf.predict(X_test)
        f.write(f"{save_metrics(y_test.to_list(), list(predicted))}\n\n")

        cm = confusion_matrix(y_test, predicted)

        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f"{dict_letter}")

        cm_filename = f"{dataset}_multiclass_cm.png"
        plt.savefig(Path("conf_matrix") / cm_filename, bbox_inches='tight')
        plt.close()

        f.write('full_test\n')
        predicted = clf.predict(X_test_full)
        f.write(f"{save_metrics(y_test_full.to_list(), list(predicted))}\n\n")


input_data_path = ['tabular_data/CIFAR10/merged_data_all_jsma.csv',
                   'tabular_data/CIFAR10GRAY/merged_data_all_jsma.csv',
                   'tabular_data/MNIST/merged_data_all_jsma.csv',
                   'tabular_data/LaVAN/merged_data.csv']


for data_path in input_data_path:
    dataset = data_path.split('/')[1]
    dict_letter = {'CIFAR10': 'a', 'CIFAR10GRAY': 'b', 'MNIST': 'c', 'LaVAN': 'd'}
    data = pd.read_csv(data_path, index_col=0, encoding="windows-1251")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=3802)
    dropped = ['flag', 'title']
    if dataset == 'CIFAR10GRAY':
        dropped.extend(['mean', 'median', 'range', 'std', '75q', '97q', '3sigma_cnt', '5sigma_cnt', '7sigma_cnt', '3iqr_cnt', '6iqr_cnt'])
    elif dataset == 'CIFAR10':
        dropped.extend(['median', 'range', 'std', 'iqr', '7sigma_cnt', '3iqr_cnt', '6iqr_cnt', 'skew'])
    elif dataset == 'MNIST':
        dropped.extend(
            ['median', 'range', 'std', '25q', '75q', '95q', '97q', '99q', 'cv', '5sigma_cnt', '3iqr_cnt', '6iqr_cnt'])
    elif dataset == 'LaVAN':
        dropped.extend(
            ['median', 'range', 'std', '25q', '75q', '95q', '97q', '99q', 'iqr', '3sigma_cnt', '7sigma_cnt', '3iqr_cnt',
             '6iqr_cnt'])
    X_train, y_train = train_data.drop(dropped, axis=1), train_data['flag']
    X_test, y_test = test_data.drop(dropped, axis=1), test_data['flag']
    data = pd.read_csv(data_path, index_col=0, encoding="windows-1251")
    X_test_full, y_test_full = data.drop(dropped, axis=1), data['flag']

    best_F = 0
    best_model = ''
    models = ['logistic', 'forest', 'svm']
    solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    kernels_one_class = ['linear', 'rbf', 'poly']
    for model in models:
        if model == 'logistic':
            for solver in solvers:
                print(f"solver:\t{solver}")
                model_name = f'{model}_{dataset}_{solver}.pkl'
                metrics_filename = f"{dataset}_{model}_{solver}.txt"
                clf = LogisticRegression(penalty='l2', solver=solver, random_state=3802)
                clf.fit(X_train, y_train)
                predicted = clf.predict(X_test)
                F = save_metrics(y_test.to_list(), list(predicted))['f1']
                if F > best_F:
                    best_F = F
                    best_model = metrics_filename
                    get_metrics(clf, X_train, X_test, X_test_full, y_train, y_test, y_test_full, metrics_filename,
                                model_name, dict_letter[dataset], dataset)
        elif model == 'forest':
            best_F = 0
            for n_estimators in range(1, 201):
                print(f"n_estimators:\t{n_estimators}")
                model_name = f'{model}_{dataset}_{n_estimators}.pkl'
                metrics_filename = f"{dataset}_{model}_{n_estimators}.txt"
                clf = RandomForestClassifier(n_estimators=n_estimators)
                clf.fit(X_train, y_train)
                predicted = clf.predict(X_test)
                F = save_metrics(y_test.to_list(), list(predicted))['f1']
                if F > best_F:
                    best_F = F
                    best_model = metrics_filename
                    get_metrics(clf, X_train, X_test, X_test_full, y_train, y_test, y_test_full, metrics_filename,
                                model_name, dict_letter[dataset], dataset)
        elif model == 'svm':
            best_F = 0
            for kernel in kernels_one_class:
                print(f"kernel:\t{kernel}")
                if kernel == 'poly':
                    for deg in range(1, 3):
                        model_name = f"{dataset}_svm_{kernel}_{deg}.pkl"
                        metrics_filename = f"{dataset}_svm_{kernel}_{deg}.txt"
                        clf = svm.SVC(kernel=kernel, degree=deg)
                        clf.fit(X_train, y_train)
                        predicted = clf.predict(X_test)
                        F = save_metrics(y_test.to_list(), list(predicted))['f1']
                        if F > best_F:
                            best_F = F
                            best_model = metrics_filename
                            get_metrics(clf, X_train, X_test, X_test_full, y_train, y_test, y_test_full,
                                        metrics_filename, model_name, dict_letter[dataset], dataset)
                else:
                    model_name = f"{dataset}_svm_{kernel}.pkl"
                    metrics_filename = f"{dataset}_svm_{kernel}.txt"
                    clf = svm.SVC(kernel=kernel)
                    clf.fit(X_train, y_train)
                    predicted = clf.predict(X_test)
                    F = save_metrics(y_test.to_list(), list(predicted))['f1']
                    if F > best_F:
                        best_F = F
                        best_model = metrics_filename
                        get_metrics(clf, X_train, X_test, X_test_full, y_train, y_test, y_test_full, metrics_filename,
                                    model_name, dict_letter[dataset], dataset)
    print(f"\n\nBest F1:\t\t{best_F}\tBest model:\t\t{best_model}")
