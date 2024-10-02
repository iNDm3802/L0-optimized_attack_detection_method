import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
from scipy.stats import kurtosis, skew
from mpire import WorkerPool

def load_text_data(text_file_path):
    with open(text_file_path, 'r') as f:
        content = f.read()
        extracted_numbers_list = content.split(';')
        data_one_image_list = list(map(float, extracted_numbers_list))
    return np.asarray(data_one_image_list)

def normalize_statistics(statistics, count_values, factor=1000):
    return [stat / count_values * factor for stat in statistics]

def process_file(file, path_to_folder="D:\zzz ImageNet+LaVAN"):
    file_data = load_text_data(os.path.join(path_to_folder, file))
    title = os.path.splitext(os.path.basename(file))[0].split(',')[0][2:-1]
    count_values = len(file_data)

    mean = np.mean(file_data)
    median = np.median(file_data)
    max_value = np.max(file_data)
    min_value = np.min(file_data)
    range_value = max_value - min_value
    var = np.var(file_data)
    std = np.std(file_data)

    quantile_75 = np.quantile(file_data, 0.75)
    quantile_25 = np.quantile(file_data, 0.25)
    iqr = quantile_75 - quantile_25

    cv = std / mean

    three_sigma_cnt = np.sum(file_data > (mean + 3 * std))
    three_sigma_cnt /= count_values
    five_sigma_cnt = np.sum(file_data > (mean + 5 * std))
    five_sigma_cnt /= count_values
    seven_sigma_cnt = np.sum(file_data > (mean + 7 * std))
    seven_sigma_cnt /= count_values

    upper_bound = quantile_75 + 1.5 * iqr
    three_iqr_cnt = np.sum(file_data > upper_bound)
    three_iqr_cnt /= count_values

    upper_bound_wide = quantile_75 + 3 * iqr
    six_iqr_cnt = np.sum(file_data > upper_bound_wide)
    six_iqr_cnt /= count_values

    skewness = skew(file_data)
    kurtosis_value = kurtosis(file_data)

    quantile_95 = np.quantile(file_data, 0.95)
    quantile_97 = np.quantile(file_data, 0.97)
    quantile_99 = np.quantile(file_data, 0.99)

    statistics = [mean, median, max_value, range_value, var, std, quantile_75, quantile_25, quantile_95, 
                  quantile_97, quantile_99, iqr, cv, three_sigma_cnt, five_sigma_cnt, seven_sigma_cnt, 
                  three_iqr_cnt, six_iqr_cnt, skewness, kurtosis_value]
    
    return [title, count_values] + statistics


def temp(x):
    return process_file(x)


def normalize(a):
    '''
    Normalizes 2D-array

    :param a: array
    :return: normalized array
    '''
    return (a - a.min()) / (a.max() - a.min())


def txt_files2csv(path_to_folder, suffix=None, num_workers=8):
    print(path_to_folder)
    if suffix == "px":
        text_files = [f for f in os.listdir(path_to_folder) if
                      f.endswith('.txt') and not "clean" in f and not "jsma" in f]
    elif suffix:
        text_files = [f for f in os.listdir(path_to_folder) if f.endswith('.txt') and suffix in f]
    else:
        text_files = [f for f in os.listdir(path_to_folder) if f.endswith('.txt')]
    file_info_list = [file for file in text_files]

    with WorkerPool(n_jobs=num_workers) as pool:
        data = list(tqdm(pool.imap_unordered(process_file, file_info_list), total=len(file_info_list)))

    columns = ['title', 'count_values', 'mean', 'median', 'max', 'range', 'var', 'std', '75q', '25q', 
               '95q', '97q', '99q', 'iqr', 'cv', '3sigma_cnt', '5sigma_cnt', '7sigma_cnt', '3iqr_cnt', 
               '6iqr_cnt', 'skew', 'kurt']
    
    df = pd.DataFrame(data, columns=columns)
    print(df)
    
    return df

def collect_data(dataset_name, dataset_path, normalize=False, num_workers=8):

    output_path = Path('tabular_data') / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    if dataset_name == 'LaVAN':
        lavan_clean_df = txt_files2csv(Path(dataset_path), suffix="original", normalize=normalize, num_workers=num_workers)
        lavan_adv_df = txt_files2csv(Path(dataset_path), suffix="adversarial", normalize=normalize, num_workers=num_workers)

        lavan_clean_df['flag'] = 0
        lavan_adv_df['flag'] = 1
        merged_data = pd.concat([lavan_clean_df, lavan_adv_df], ignore_index=True).drop(['count_values'], axis=1)
        merged_data_cleaned = merged_data.drop(['mean', '75q', '25q', '95q', 'cv', '3iqr_cnt'], axis=1)

        merged_data_cleaned = merged_data_cleaned.drop(merged_data_cleaned.loc[merged_data_cleaned['median'].isna()].index).reset_index(drop=True)
        merged_data = merged_data.drop(merged_data.loc[merged_data['median'].isna()].index).reset_index(drop=True)
        
        merged_data.to_csv(output_path / 'merged_data.csv', index=False)
        merged_data_cleaned.to_csv(output_path / 'merged_data_cleaned.csv', index=False)
    else:
        clean_df = txt_files2csv(Path(dataset_path), suffix="clean", normalize=normalize, num_workers=num_workers)
        px_df = txt_files2csv(Path(dataset_path), suffix="px", normalize=normalize, num_workers=num_workers)
        jsma_df = txt_files2csv(Path(dataset_path), suffix="jsma", normalize=normalize, num_workers=num_workers)
        
        clean_df['flag'] = 0
        px_df['flag'] = 1
        jsma_df['flag'] = 2

        merged_data_all_jsma = pd.concat([clean_df, px_df, jsma_df], ignore_index=True).drop(
            ['count_values'], axis=1)

        merged_data_all_jsma = merged_data_all_jsma.drop(merged_data_all_jsma.loc[merged_data_all_jsma['median'].isna()].index).reset_index(drop=True)
        merged_data_all_jsma.to_csv(output_path / 'merged_data_all_jsma.csv', index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Collect tabular data from datasets.')
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help='The path to the dataset folder')
    parser.add_argument('--normalize', action='store_true', help='Normalize statistics to be independent of image size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for multiprocessing')

    args = parser.parse_args()
    collect_data(args.dataset, args.dataset_path, args.normalize, args.num_workers)
