import os
import pandas as pd
import json
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import scipy.stats as ss


PROJECT_PATH = os.path.abspath(os.pardir)
OSTEOPENIA_DATASET = os.path.join(PROJECT_PATH, "osteopenia_dataset")
CSV_PATH_OSTEOPENIA = os.path.join(OSTEOPENIA_DATASET, "osteopenia_dataset.csv")

data = []

def isNaN(num):
    return num != num

# https://stackoverflow.com/questions/46498455/categorical-features-correlation/46498792#46498792
def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.to_numpy().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

#for each patient, patient ids are folder names
classifications = {}
num_of_classes = {}
osteopenia_patient_ids = []
for patient in os.listdir(OSTEOPENIA_DATASET):

    #ignore csv file
    if patient == "osteopenia_dataset.csv":
        continue

    osteopenia_patient_ids.append(patient)
    #get per patient labels
    patient_lbls = []
    patient_dir = os.path.join(OSTEOPENIA_DATASET, patient)
    for file in os.listdir(patient_dir):

        name, ext = os.path.splitext(file)
        if ext == '.json':
            file_path = os.path.join(patient_dir, file)
            with open(file_path, 'r') as labels:
                lbl_dict = json.load(labels)
            patient_lbls.append(lbl_dict)
        else:
            continue

    #sort patient lbls (maybe for future use)
    sorted_lbls = sorted(patient_lbls, key=lambda d: d['study_number'])

    #prepare labels for statistical testing (nan == 0), (M == 1), (F == 0)
    for patient_lbl in sorted_lbls:
        if np.isnan(patient_lbl['osteopenia']):
            patient_lbl['osteopenia'] = 0
        if np.isnan(patient_lbl['initial_exam']):
            patient_lbl['initial_exam'] = 0
        if np.isnan(patient_lbl['fracture_visible']):
            patient_lbl['fracture_visible'] = 0
        if np.isnan(patient_lbl['metal']):
            patient_lbl['metal'] = 0
        if np.isnan(patient_lbl['cast']):
            patient_lbl['cast'] = 0
        if patient_lbl['gender'] == 'M':
            patient_lbl['gender'] = 1
        else:
            patient_lbl['gender'] = 0
        data.append([patient_lbl['osteopenia'], patient_lbl['gender'], patient_lbl['study_number'], patient_lbl['age'], patient_lbl['initial_exam'], patient_lbl['fracture_visible'], patient_lbl['metal'], patient_lbl['cast']])

        if not isNaN(patient_lbl['ao_classification']):
            classes = patient_lbl['ao_classification'].split(";")

            count = len(classes)
            if count in num_of_classes:
                num_of_classes[count] = num_of_classes[count] + 1
            else:
                num_of_classes[count] = 1

            for c in classes:
                cls = c.strip()
                if cls in classifications:
                    classifications[cls] = classifications[cls] + 1
                else:
                    classifications[cls] = 1

#create patients dataframe
patients_df = pd.DataFrame(data, columns=["osteopenia", 'gender', "study_number", "age", "initial_exam", "fracture_visible", "metal", "cast"])


#plot histogram of diagnoses for patients with osteopenia (some have more diagnoses)
classifications = sorted(classifications.items(), key=lambda x:x[1], reverse=True)
plt.figure(figsize=(8, 8))
xs = [x for x,y in classifications]
ys = [y for x,y in classifications]
sum_ys = sum(ys)
plt.bar(xs, np.array(ys)/sum_ys, width=1)
plt.xticks(rotation='vertical')
plt.title("Distribution of diagnoses for patients with osteopenia")
plt.savefig("plots/diagnose_hist_with_osteopenia.png")

num_of_classes = sorted(num_of_classes.items(), key=lambda x:x[1], reverse=True)
plt.figure(figsize=(8, 10))
xx = [x for x,y in num_of_classes]
yy = [y for x,y in num_of_classes]
sum_yy = sum(yy)
plt.bar(xx, np.array(yy)/sum_yy, width=0.1)
plt.xticks(rotation='vertical')
plt.title("Distribution of fractures count for patients with osteopenia")
plt.savefig("plots/fracture_count_hist_with_osteopenia.png")


#https://www.statology.org/interpret-cramers-v/
#cramers V [0,1]
#categorical-categorical: osteopenia, gender, fracture visible, metal, initial_exam
print("--------------------------- Categorical - categorical variable relationship ---------------------------")
cm_osteopenia_gender = pd.crosstab(patients_df['osteopenia'], patients_df['gender'])
print("[Osteopenia vs gender]",cramers_v(cm_osteopenia_gender), ", p-value", ss.chi2_contingency(cm_osteopenia_gender)[1])#none
cm_osteopenia_fracture = pd.crosstab(patients_df['osteopenia'], patients_df['fracture_visible'])
print("[Osteopenia vs fracture_visible]",cramers_v(cm_osteopenia_fracture), ", p-value", ss.chi2_contingency(cm_osteopenia_fracture)[1])#none
cm_osteopenia_metal = pd.crosstab(patients_df['osteopenia'], patients_df['metal'])
print("[Osteopenia vs metal]",cramers_v(cm_osteopenia_metal), ", p-value", ss.chi2_contingency(cm_osteopenia_metal)[1])#none
cm_osteopenia_initial_exam = pd.crosstab(patients_df['osteopenia'], patients_df['initial_exam'])
print("[Osteopenia vs initial_exam]",cramers_v(cm_osteopenia_initial_exam), ", p-value", ss.chi2_contingency(cm_osteopenia_initial_exam)[1]) #small-to-medium, significant
cm_osteopenia_cast = pd.crosstab(patients_df['osteopenia'], patients_df['cast'])
print("[Osteopenia vs cast]",cramers_v(cm_osteopenia_initial_exam), ", p-value", ss.chi2_contingency(cm_osteopenia_cast)[1])#small-to-medium, significant

print("\n")
#point biserial [-1, 1]
#categorical-continous: osteopenia, study_number. age
print("--------------------------- Categorical - continous variable relationship ---------------------------")
print("[Osteopenia vs study_number]", ss.pointbiserialr(patients_df['osteopenia'], patients_df['study_number'])) #significant positive correlation
print("[Osteopenia vs age] ", ss.pointbiserialr(patients_df['osteopenia'], patients_df['age'])) #not significant


#cross-check of statistical tests
plt.figure(figsize=(8, 8))
sn.heatmap(patients_df.corr(), annot=True)
plt.savefig("plots/corr_matrix_with_osteopenia.png")


#non osteopenians distribution
CSV_PATH_ALL = os.path.join(PROJECT_PATH, "dataset", "dataset.csv")

df = pd.read_csv(CSV_PATH_ALL)

patients_without_osteopenia_diag = {}
patients_without_osteopenia_count_frac = {}
for i in range(len(df)):

    if df.iloc[i]['patient_id'] in osteopenia_patient_ids:
        continue

    ao_classification = df.iloc[i]['ao_classification']
    if not isNaN(ao_classification):
        classes = ao_classification.split(";")

        count = len(classes)
        if count in patients_without_osteopenia_count_frac:
            patients_without_osteopenia_count_frac[count] = patients_without_osteopenia_count_frac[count] + 1
        else:
            patients_without_osteopenia_count_frac[count] = 1

        for c in classes:
            cls = c.strip()
            if cls in patients_without_osteopenia_diag:
                patients_without_osteopenia_diag[cls] = patients_without_osteopenia_diag[cls] + 1
            else:
                patients_without_osteopenia_diag[cls] = 1


#plot histogram of diagnoses for patients without osteopenia (some have more diagnoses)
patients_without_osteopenia_diag = sorted(patients_without_osteopenia_diag.items(), key=lambda x:x[1], reverse=True)
plt.figure(figsize=(12, 12))
xs = [x for x,y in patients_without_osteopenia_diag]
ys = [y for x,y in patients_without_osteopenia_diag]
sum_ys = sum(ys)
plt.bar(xs, np.array(ys)/sum_ys, width=1)
plt.xticks(rotation='vertical')
plt.title("Distribution of diagnoses for patients without osteopenia")
plt.savefig("plots/diagnose_hist_without_osteopenia.png")

patients_without_osteopenia_count_frac = sorted(patients_without_osteopenia_count_frac.items(), key=lambda x:x[1], reverse=True)
plt.figure(figsize=(8, 8))
xx = [x for x,y in patients_without_osteopenia_count_frac]
yy = [y for x,y in patients_without_osteopenia_count_frac]
sum_yy = sum(yy)
plt.bar(xx, np.array(yy)/sum_yy, width=0.1)
plt.xticks(rotation='vertical')
plt.title("Distribution of fractures count for patients without osteopenia")
plt.savefig("plots/fracture_count_hist_without_osteopenia.png")
