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


def get_unique_diagnoses_and_patient_ids(data_frame):
    patient_ids = OrderedSet()
    diagnoses = OrderedSet()

    for i in range(len(data_frame)):
        patient_ids.add(data_frame.iloc[i]["patient_id"])
        if str(data_frame.iloc[i]["ao_classification"]) != "0":
            classes = str(data_frame.iloc[i]["ao_classification"]).split(";")
            for c in classes:
                if c[-1] != ".":
                    diagnoses.add(c.strip())

    return patient_ids, diagnoses


def get_osteopenia_per_patient_df(data_frame, patient_ids):
    column_osteopenia = []

    for patient_id in patient_ids:
        patient_df = data_frame.loc[data_frame["patient_id"] == patient_id]
        if len(patient_df.loc[patient_df["osteopenia"] == 1]) >= 1:
            column_osteopenia.append(1)
        else:
            column_osteopenia.append(0)

    return pd.DataFrame({'osteopenia': column_osteopenia})


def main():
    data_frame = pd.read_csv(CSV_PATH_DATASET)
    data_frame["ao_classification"] = data_frame["ao_classification"].fillna(0)
    data_frame.sort_values("patient_id")

    patient_ids, diagnoses = get_unique_diagnoses_and_patient_ids(data_frame)
    osteopenia_df = get_osteopenia_per_patient_df(data_frame, patient_ids)

    for diagnosis in diagnoses:
        column_diagnosis = []

        # For every patient check if diagnosis is present
        for patient_id in patient_ids:
            df = data_frame.loc[data_frame["patient_id"] == patient_id]
            diagnosis_present = False

            # If the diagnosis is present in any of the patient entries
            for i in range(len(df)):
                if diagnosis in str(df.iloc[i]["ao_classification"]):
                    column_diagnosis.append(1)
                    diagnosis_present = True
                    break

            if not diagnosis_present:
                column_diagnosis.append(0)

        # Calculate correlation matrix
        corr_matrix = pd.crosstab(osteopenia_df['osteopenia'],
                                  pd.DataFrame({diagnosis: column_diagnosis})[diagnosis])
        cramer = cramers_v(corr_matrix)
        p_value = ss.chi2_contingency(corr_matrix)[1]

        # if the correlation isn't significant
        if p_value > 0.05 or cramer < 0.1:
            continue

        if 0.1 <= cramer < 0.3:
            correlation_text = "Small"
        elif 0.3 <= cramer < 0.5:
            correlation_text = "Medium"
        else:
            correlation_text = "Large"

        print(f"Osteopenia vs {diagnosis}: {cramer} ({correlation_text} Correlation), p-value: {p_value}, "
              f"diagnosis count: {np.sum(column_diagnosis)}")

        correlation_df = pd.DataFrame(data=[[1.0, cramer], [cramer, 1.0]], index=['osteopenia', diagnosis],
                                      columns=['osteopenia', diagnosis])

        # Plot correlation Matrix
        plt.figure()
        plt.title(diagnosis + " osteopenia_corr_matrix")
        sn.heatmap(correlation_df, annot=True, xticklabels=True, yticklabels=True)
        plt.savefig(
            f"plots/diagnoses_correlation_plots/osteopenia_diagnosis{diagnosis.replace('/', '-')}_corr_matrix.png")


if __name__ == "__main__":
    main()
