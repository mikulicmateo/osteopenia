import os
import pandas as pd
import math
import seaborn as sn
import matplotlib.pyplot as plt

PROJECT_PATH = os.path.abspath(os.pardir)
CSV_PATH_DATASET = os.path.join(PROJECT_PATH, "dataset/dataset.csv")
CSV_PATH_OSTEOPENIA = os.path.join(PROJECT_PATH, "osteopenia_dataset/osteopenia_dataset.csv")
CSV_PATHS = [CSV_PATH_DATASET, CSV_PATH_OSTEOPENIA]
corr_matrix_names = ["osteopenia_dataset_corr_matrix", "dataset_corr_matrix"]


for i in range(2):
    data_frame = pd.read_csv(CSV_PATHS[i])

    data_frame = data_frame.drop(['filestem'], axis=1)
    data_frame = data_frame.drop(['patient_id'], axis=1)
    data_frame = data_frame.drop(['pixel_spacing'], axis=1)
    data_frame = data_frame.drop(['device_manufacturer'], axis=1)
    data_frame = data_frame.drop(['ao_classification'], axis=1)

    for j in range(len(data_frame)):
        if data_frame.iloc[j]['gender'] == 'M':
            data_frame.at[j, 'gender'] = 1
        else:
            data_frame.at[j, 'gender'] = 0

        if data_frame.iloc[j]['laterality'] == 'L':
            data_frame.at[j, 'laterality'] = 1
        else:
            data_frame.at[j, 'laterality'] = 0

        for column in data_frame.columns:
            if math.isnan(data_frame.iloc[j][column]):
                data_frame.at[j, column] = 0

    plt.figure(figsize=(15, 12))
    plt.title(corr_matrix_names[i])
    sn.heatmap(data_frame.corr(), annot=True, xticklabels=True, yticklabels=True)
    plt.savefig(corr_matrix_names[i] + ".png")
