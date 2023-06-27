import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sn
from sortedcollections import OrderedSet

from statistical_tests_distributions import cramers_v

PROJECT_PATH = os.path.abspath(os.pardir)
CSV_PATH_DATASET = os.path.join(PROJECT_PATH, "dataset/dataset.csv")
patient_ids = OrderedSet()
diagnoses = OrderedSet()


def main():
    data_frame = pd.read_csv(CSV_PATH_DATASET)
    data_frame["ao_classification"] = data_frame["ao_classification"].fillna(0)

    for i in range(len(data_frame)):
        patient_ids.add(data_frame.iloc[i]["patient_id"])
        if str(data_frame.iloc[i]["ao_classification"]) != "0":
            classes = str(data_frame.iloc[i]["ao_classification"]).split(";")
            for c in classes:
                diagnoses.add(c.strip())

    correlation_dicts = []
    image_name_counter = 0

    for diagnosis in diagnoses:
        column_diagnosis = []
        column_osteopenia = []

        for patient_id in patient_ids:
            df = data_frame.loc[data_frame["patient_id"] == patient_id]
            df.sort_values('timehash')

            for i in range(len(df)):

                if str(df.iloc[i]["ao_classification"]) != "0":
                    if diagnosis in str(df.iloc[i]["ao_classification"]):
                        column_diagnosis.append(1)
                    else:
                        column_diagnosis.append(0)
                else:
                    column_diagnosis.append(0)

                if len(df.loc[data_frame["osteopenia"] == 1]) >= 1:
                    column_osteopenia.append(1)
                else:
                    column_osteopenia.append(0)

                break

        diagnosis_df = pd.DataFrame({diagnosis: column_diagnosis})
        osteopenia_df = pd.DataFrame({'osteopenia': column_osteopenia})
        corr_matrix = pd.crosstab(osteopenia_df['osteopenia'], diagnosis_df[diagnosis])
        cramer = cramers_v(corr_matrix)
        p_value = ss.chi2_contingency(corr_matrix)[1]
        if p_value <= 0.05 and cramer >= 0.1:
            if 0.1 <= cramer < 0.3:
                correlation_text = "Small"
            elif 0.3 <= cramer < 0.5:
                correlation_text = "Medium"
            else:
                correlation_text = "Large"

            corr_dict = {
                "diagnosis": diagnosis,
                "cramer": cramer,
                "correlation_interpretation": correlation_text,
                "p-value": p_value,
                "num_of_diagnoses": np.sum(column_diagnosis)
            }

            correlation_dicts.append(corr_dict)

            temp_df = pd.DataFrame(data=[[1.0, cramer], [cramer, 1.0]], index=['osteopenia', diagnosis],
                                   columns=['osteopenia', diagnosis])

            plt.figure()
            plt.title(diagnosis + " osteopenia_corr_matrix")
            sn.heatmap(temp_df, annot=True, xticklabels=True, yticklabels=True)
            plt.savefig(f"plots/diagnoses_correlation_plots/osteopenia_diagnosis{image_name_counter}_corr_matrix.png")
            image_name_counter += 1

    for corr_dict in correlation_dicts:
        print(
            f"Osteopenia vs {corr_dict['diagnosis']}: {corr_dict['cramer']} "
            f"({corr_dict['correlation_interpretation']} Correlation), p-value: {corr_dict['p-value']}, "
            f"diagnosis count: {corr_dict['num_of_diagnoses']}")


if __name__ == "__main__":
    main()
