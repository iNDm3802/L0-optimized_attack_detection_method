from preprocessing import compute_score
from scipy.stats import kurtosis, skew

import cv2
import numpy as np
import pickle
import time

import warnings


warnings.filterwarnings("ignore")


def compute_stat(score, anomaly=True, color=True, gray=False, bw=False, lavan=False):

    # if dataset == 'CIFAR10GRAY':  # 54
    #     dropped.extend(['mean', 'std', '75q', '95q', '97q', '99q', 'iqr', '3iqr_cnt', '6iqr_cnt', 'skew'])
    # elif dataset == 'CIFAR10':  # 9
    #     dropped.extend(['median', 'var', '25q', '97q', 'iqr', '3sigma_cnt', '3iqr_cnt', '6iqr_cnt', 'kurt'])
    # elif dataset == 'MNIST':  # 23
    #     dropped.extend(['75q', '95q', '5sigma_cnt', '3iqr_cnt', '6iqr_cnt'])
    # elif dataset == 'LaVAN':
    #     dropped.extend(['iqr'])
    if anomaly:
        count_values = len(score)
        if color:
            mean = np.mean(score)
            max_value = np.max(score)
            min_value = np.min(score)
            range_value = max_value - min_value
            # var = np.var(score)
            std = np.std(score)

            cv = std / mean

            quantile_75 = np.quantile(score, 0.75)
            # quantile_25 = np.quantile(score, 0.25)
            quantile_95 = np.quantile(score, 0.95)
            # quantile_97 = np.quantile(score, 0.97)
            quantile_99 = np.quantile(score, 0.99)

            five_sigma_cnt = np.sum(score > (mean + 5 * std))
            five_sigma_cnt /= count_values
            seven_sigma_cnt = np.sum(score > (mean + 7 * std))
            seven_sigma_cnt /= count_values

            skewness = skew(score)

            statistics = [mean, max_value, range_value, std,
                          quantile_75, quantile_95, quantile_99, cv,
                          five_sigma_cnt, seven_sigma_cnt, skewness]
        elif gray:
            mean = np.mean(score)
            median = np.median(score)
            max_value = np.max(score)
            min_value = np.min(score)
            range_value = max_value - min_value
            var = np.var(score)
            std = np.std(score)

            cv = std / mean

            quantile_25 = np.quantile(score, 0.25)

            three_sigma_cnt = np.sum(score > (mean + 3 * std))
            three_sigma_cnt /= count_values
            five_sigma_cnt = np.sum(score > (mean + 5 * std))
            five_sigma_cnt /= count_values
            seven_sigma_cnt = np.sum(score > (mean + 7 * std))
            seven_sigma_cnt /= count_values

            # skewness = skew(score)
            kurtosis_value = kurtosis(score)

            statistics = [median, max_value, range_value, var, quantile_25, cv,
                          three_sigma_cnt, five_sigma_cnt, seven_sigma_cnt, kurtosis_value]
        elif bw:
            mean = np.mean(score)
            median = np.median(score)
            max_value = np.max(score)
            min_value = np.min(score)
            range_value = max_value - min_value
            var = np.var(score)
            std = np.std(score)

            cv = std / mean

            quantile_75 = np.quantile(score, 0.75)
            quantile_25 = np.quantile(score, 0.25)
            # quantile_95 = np.quantile(score, 0.95)
            quantile_97 = np.quantile(score, 0.97)
            quantile_99 = np.quantile(score, 0.99)

            iqr = quantile_75 - quantile_25

            three_sigma_cnt = np.sum(score > (mean + 3 * std))
            three_sigma_cnt /= count_values
            seven_sigma_cnt = np.sum(score > (mean + 7 * std))
            seven_sigma_cnt /= count_values

            skewness = skew(score)
            kurtosis_value = kurtosis(score)

            statistics = [mean, median, max_value, range_value, var, std,
                          quantile_25, quantile_97, quantile_99, iqr, cv,
                          three_sigma_cnt, seven_sigma_cnt, skewness, kurtosis_value]
        # elif lavan:
        #     mean = np.mean(score)
        #     median = np.median(score)
        #     max_value = np.max(score)
        #     min_value = np.min(score)
        #     range_value = max_value - min_value
        #     var = np.var(score)
        #     std = np.std(score)
        #
        #     quantile_75 = np.quantile(score, 0.75)
        #     quantile_25 = np.quantile(score, 0.25)
        #     iqr = quantile_75 - quantile_25
        #
        #     cv = std / mean
        #
        #     three_sigma_cnt = np.sum(score > (mean + 3 * std))
        #     three_sigma_cnt /= count_values
        #     five_sigma_cnt = np.sum(score > (mean + 5 * std))
        #     five_sigma_cnt /= count_values
        #     seven_sigma_cnt = np.sum(score > (mean + 7 * std))
        #     seven_sigma_cnt /= count_values
        #
        #     upper_bound = quantile_75 + 1.5 * iqr
        #     three_iqr_cnt = np.sum(score > upper_bound)
        #     three_iqr_cnt /= count_values
        #
        #     upper_bound_wide = quantile_75 + 3 * iqr
        #     six_iqr_cnt = np.sum(score > upper_bound_wide)
        #     six_iqr_cnt /= count_values
        #
        #     skewness = skew(score)
        #     kurtosis_value = kurtosis(score)
        #
        #     quantile_95 = np.quantile(score, 0.95)
        #     quantile_97 = np.quantile(score, 0.97)
        #     quantile_99 = np.quantile(score, 0.99)
        #
        #     statistics = [mean, median, max_value, range_value, var, std, quantile_75, quantile_25, quantile_95,
        #                   quantile_97, quantile_99, cv, three_sigma_cnt, five_sigma_cnt, seven_sigma_cnt,
        #                   three_iqr_cnt, six_iqr_cnt, skewness, kurtosis_value]
    elif anomaly and lavan or not anomaly:
        count_values = len(score)

        mean = np.mean(score)
        median = np.median(score)
        max_value = np.max(score)
        min_value = np.min(score)
        range_value = max_value - min_value
        var = np.var(score)
        std = np.std(score)

        quantile_75 = np.quantile(score, 0.75)
        quantile_25 = np.quantile(score, 0.25)
        iqr = quantile_75 - quantile_25

        cv = std / mean

        three_sigma_cnt = np.sum(score > (mean + 3 * std))
        three_sigma_cnt /= count_values
        five_sigma_cnt = np.sum(score > (mean + 5 * std))
        five_sigma_cnt /= count_values
        seven_sigma_cnt = np.sum(score > (mean + 7 * std))
        seven_sigma_cnt /= count_values

        upper_bound = quantile_75 + 1.5 * iqr
        three_iqr_cnt = np.sum(score > upper_bound)
        three_iqr_cnt /= count_values

        upper_bound_wide = quantile_75 + 3 * iqr
        six_iqr_cnt = np.sum(score > upper_bound_wide)
        six_iqr_cnt /= count_values

        skewness = skew(score)
        kurtosis_value = kurtosis(score)

        quantile_95 = np.quantile(score, 0.95)
        quantile_97 = np.quantile(score, 0.97)
        quantile_99 = np.quantile(score, 0.99)

        statistics = [mean, median, max_value, range_value, var, std, quantile_75, quantile_25, quantile_95,
                      quantile_97, quantile_99, iqr, cv, three_sigma_cnt, five_sigma_cnt, seven_sigma_cnt,
                      three_iqr_cnt, six_iqr_cnt, skewness, kurtosis_value]

    return statistics


count = 10000

# img_path = 'images\CIFAR10\\a_0.png'
# clfs = ['models/!_anomaly_CUT/SVM_CIFAR10.pkl',
#         'models/!_multiclass_FULL/forest_CIFAR10_multi.pkl',
#         'models/!_binary_FULL/forest_CIFAR10_binary.pkl']

# img_path = 'images\CIFAR10G\\a_0.png'
# clfs = ['models/!_anomaly_CUT/forest_CIFAR10GRAY.pkl',
#         'models/!_multiclass_FULL/forest_CIFAR10GRAY_multi.pkl',
#         'models/!_binary_FULL/forest_CIFAR10GRAY_binary.pkl']

img_path = 'images\MNIST\\b_0.png'
clfs = ['models/!_anomaly_CUT/forest_MNIST.pkl',
        'models/!_multiclass_FULL/forest_MNIST_multi.pkl',
        'models/!_binary_FULL/forest_MNIST_binary.pkl']

# img_path = 'images\ImageNet\\7655_124_adversarial.png'
# clfs = ['models/!_anomaly_FULL/forest_LaVAN.pkl']

for clf_path in clfs:
    print(clf_path)
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # gray

    clf = pickle.load(open(clf_path, 'rb'))

    start = time.time()
    for i in range(count + 1):
        score = compute_score(img).flatten()
        stat = compute_stat(score, 'anomaly' in clf_path,
                            '_CIFAR10_' in clf_path or '_CIFAR10.pkl' in clf_path,
                            'GRAY' in clf_path,
                            'MNIST' in clf_path,
                            'LaVAN' in clf_path)
        stat = np.array(stat).reshape(1, -1)
        clf.predict(stat)
        # print(f"Processing:\t\t{i/count}")
    duratiuon = time.time() - start
    print(f"Total duration for {count} images:\t\t{duratiuon} seconds")
