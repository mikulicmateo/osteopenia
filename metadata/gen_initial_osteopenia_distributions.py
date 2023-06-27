import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_PATH = os.path.abspath(os.pardir)
CSV_PATH_OSTEOPENIA = os.path.join(PROJECT_PATH, "osteopenia_dataset/osteopenia_dataset.csv")


def main():
    df = pd.read_csv(CSV_PATH_OSTEOPENIA)

    # Percentage of people who had osteopenia initially
    initial_osteopenia_count = 0
    patient_ids = set()

    for i in range(len(df)):
        patient_ids.add(df.iloc[i]["patient_id"])

    for id_patient in patient_ids:
        patient_df = df.loc[df["patient_id"] == id_patient]
        patient_df.sort_values("timehash")
        if not np.isnan(patient_df.iloc[0]["osteopenia"]):
            initial_osteopenia_count += 1

    initial_osteopenia_percentage = np.round(initial_osteopenia_count / len(patient_ids), 3)

    initial_osteopenia_dict = {
        "initial_osteopenia": initial_osteopenia_percentage * 100,
        "not_initial_osteopenia": (1 - initial_osteopenia_percentage) * 100,
    }
    plt.title("Initial Osteopenia Distribution")
    plt.bar(initial_osteopenia_dict.keys(), initial_osteopenia_dict.values())
    plt.ylabel("Percentage")
    plt.savefig("plots/initial_osteopenia_distribution.png")


if __name__ == "__main__":
    main()
