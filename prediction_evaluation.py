import json
import os
from util.train_util import *
from model.models import *
from trainer.trainer import Trainer
from data.OsteopeniaDataset import OsteopeniaDataset
from torch import nn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter
from collections import defaultdict


def plot_missclassifications(indices_m0, indices_m1, title='frac'):
    df = pd.read_csv(os.path.join(os.getcwd(), "data", "test_dataset.csv"))
    paths = df['filestem'].to_numpy()
    misclassified_0 = [paths[k] for k in indices_m0[-8:]]
    misclassified_1 = [paths[k] for k in indices_m1[-8:]]

    f, axarr = plt.subplots((len(misclassified_0) // 2), 2, figsize=(15, 15))
    f.suptitle("False Positives", fontsize=30)

    row = 0
    column = 0
    for i in range(len(misclassified_0)):
        image = plt.imread(misclassified_0[i])
        axarr[row, column].imshow(image, cmap='gray')
        axarr[row, column].axis('off')
        if i % 2 == 0:
            if column == 1:
                column = 0
            else:
                column = 1
        else:
            row += 1

    for ax in f.axes:
        ax.axison = False
    plt.savefig(f"{title}_missclasified_0.png")

    f, axarr = plt.subplots((len(misclassified_1) // 2) + 1, 2, figsize=(20, 20))
    f.suptitle("False Negatives", fontsize=30)

    row = 0
    column = 0
    for i in range(len(misclassified_1)):
        image = plt.imread(misclassified_1[i])
        axarr[row, column].imshow(image, cmap='gray')
        if i % 2 == 0:
            if column == 1:
                column = 0
            else:
                column = 1
        else:
            row += 1

    for ax in f.axes:
        ax.axison = False
    plt.savefig(f"{title}_missclasified_1.png")


def get_name_from_path(path):
    name = path.split("/")[-1].split(".")[0]
    return name


def remove_fracture(name, img):
    with open(os.path.join(config_dict['additional_annotations_path'], "test", f"{name}.json")) as file:
        d = json.load(file)

    for object in d["objects"]:
        title = object.get("classTitle")
        if title == "fracture":
            points = object["points"]["exterior"]
            img[int(points[0][1]): int(points[1][1]), int(points[0][0]): int(points[1][0])] = 0

    return img


def plot_tp_and_tn(tp_ind_with_perc, tn_ind_with_perc, nofrac_ds):
    df = pd.read_csv(os.path.join(os.getcwd(), "data", "test_dataset.csv"))
    paths = df['filestem'].to_numpy()
    idx_tp, percs_tp = map(list, zip(*tp_ind_with_perc))
    idx_tn, percs_tn = map(list, zip(*tn_ind_with_perc))
    percs_nofrac_tp = []
    percs_nofrac_tn = []
    trainer.test_dl = trainer.create_dataloader(nofrac_ds, 1, 4, False)

    loop = tqdm(trainer.test_dl, leave=False, unit="batch", mininterval=0)
    for i, data in enumerate(loop):
        image, lbl = data
        if i in idx_tp[:8]:
            percs_nofrac_tp.append(trainer.predict(image).detach().cpu().numpy()[0, 0])
        elif i in idx_tn[:8]:
            percs_nofrac_tn.append(trainer.predict(image).detach().cpu().numpy()[0, 0])

    f, axarr = plt.subplots(4, 2, figsize=(6, 12))
    f.suptitle("TP Frac vs ? NoFrac", fontsize=15)

    row = 0
    for i, j in enumerate(idx_tp[:8]):
        image = plt.imread(paths[j])
        nofrac_img = plt.imread(paths[j])

        name = get_name_from_path(paths[j])
        nofrac_img = remove_fracture(name, nofrac_img)

        axarr[row, 0].imshow(image, cmap='gray')
        axarr[row, 0].axis('off')
        axarr[row, 0].title.set_text(percs_tp[i])
        axarr[row, 1].imshow(nofrac_img, cmap='gray')
        axarr[row, 1].axis('off')
        axarr[row, 1].title.set_text(percs_nofrac_tp[i])

        if i % 2 != 0:
            row += 1

    for ax in f.axes:
        ax.axison = False

    plt.tight_layout()
    plt.savefig(f"tp_ok.png")

    f, axarr = plt.subplots(4, 2, figsize=(6, 12))
    f.suptitle("TN Frac vs ? NoFrac", fontsize=15)

    row = 0
    for i, j in enumerate(idx_tn[:8]):
        image = plt.imread(paths[j])
        nofrac_img = plt.imread(paths[j])

        name = get_name_from_path(paths[j])
        nofrac_img = remove_fracture(name, nofrac_img)

        axarr[row, 0].imshow(image, cmap='gray')
        axarr[row, 0].axis('off')
        axarr[row, 0].title.set_text(percs_tn[i])
        axarr[row, 1].imshow(nofrac_img, cmap='gray')
        axarr[row, 1].axis('off')
        axarr[row, 1].title.set_text(percs_nofrac_tn[i])

        if i % 2 != 0:
            row += 1

    for ax in f.axes:
        ax.axison = False

    plt.tight_layout()
    plt.savefig(f"tn_ok.png")


def calculate_test_metrics(trainer, dataset):
    trainer.test_dl = trainer.create_dataloader(dataset, 1, 4, False)

    loop = tqdm(trainer.test_dl, leave=False, unit="batch", mininterval=0)

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    indices_m0 = []
    indices_m1 = []
    indices_and_perc_t0 = []
    indices_and_perc_t1 = []

    k = 0
    for image, label in loop:
        prediction = trainer.predict(image).detach().cpu().numpy()[0, 0]
        if prediction >= 0.5:
            if label == 1:
                tp += 1
                indices_and_perc_t1.append([k, prediction])
            else:
                fp += 1
                indices_m0.append(k)
        else:
            if label == 1:
                fn += 1
                indices_m1.append(k)
            else:
                tn += 1
                indices_and_perc_t0.append([k, prediction])

        k += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * (precision * recall) / (precision + recall)

    metrics = defaultdict(list)
    metrics['accuraccy'].append((tp + tn) / (tp + fp + fn + tn))
    metrics['f1'].append(f1)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['tpp'].append(tp / (tp + fp + fn + tn))
    metrics['fpp'].append(fp / (tp + fp + fn + tn))
    metrics['tnp'].append(tn / (tp + fp + fn + tn))
    metrics['fnp'].append(fn / (tp + fp + fn + tn))

    return metrics, indices_m0, indices_m1, indices_and_perc_t1, indices_and_perc_t0


config_file_path = os.path.join(os.getcwd(), 'config.json')
stage = 9

with open(config_file_path, "r") as config_file:
    config_dict = json.load(config_file)

m = choose_model(config_dict["model_name"], False)
o = torch.optim.AdamW(m.parameters(), lr=config_dict["learning_rate"], weight_decay=1)

if stage == -1:
    model_path = f"model/trained/best-{config_dict['model_name']}.pt"
else:
    model_path = f"model/trained/stage-{stage}/best-{config_dict['model_name']}.pt"

m, o, m_e = load_model(os.path.join(os.getcwd(), model_path), m,
                       o, config_dict["device"], config_dict["device_ids"])

if stage == -1:
    test_output_dir = "model/results/test/"
else:
    test_output_dir = f"model/results/test/stage-{stage}/"

trainer = Trainer(
    config_dict,
    m,
    config_dict["model_name"],
    o,
    nn.BCELoss(),
    m_e,
    test_output_dir,
    True
)

data_folder_path = os.path.join(os.getcwd(), 'data')

################## no frac
nofrac_data = OsteopeniaDataset(
    os.path.join(data_folder_path, 'test_dataset.csv'),
    config_dict['mean'],
    config_dict['std'],
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    config_dict['desired_image_size'],
    config_dict['additional_annotations_path'],
    remove_fractures=True
)

nofrac_metrics, nofrac_m0, nofrac_m1, nofrac_tp, nofrac_tn = calculate_test_metrics(trainer, nofrac_data)

print(" - - - - NOFRAC SCORES - - - - ")
print(f"accuracy = {nofrac_metrics['accuraccy']}")
print(f"F1-score = {nofrac_metrics['f1']}")
print(f"precision = {nofrac_metrics['precision']}, recall = {nofrac_metrics['recall']}")
print(" - - - - - - - - - - - - ")
print(f"tp percentage = {nofrac_metrics['tpp']}")
print(f"fp percentage = {nofrac_metrics['fpp']}")
print(f"tn percentage = {nofrac_metrics['tnp']}")
print(f"fn percentage = {nofrac_metrics['fnp']}")
print(" - - - - - - - - - - - - ")

plot_missclassifications(nofrac_m0, nofrac_m1, title='nofrac')
#################################################################
###################### frac

frac_data = OsteopeniaDataset(
    os.path.join(data_folder_path, 'test_dataset.csv'),
    config_dict['mean'],
    config_dict['std'],
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    config_dict['desired_image_size'],
)

frac_metrics, frac_m0, frac_m1, frac_tp, frac_tn = calculate_test_metrics(trainer, frac_data)

print(" - - - - FRAC SCORES - - - - ")
print(f"accuracy = {frac_metrics['accuraccy']}")
print(f"F1-score = {frac_metrics['f1']}")
print(f"precision = {frac_metrics['precision']}, recall = {nofrac_metrics['recall']}")
print(" - - - - - - - - - - - - ")
print(f"tp percentage = {frac_metrics['tpp']}")
print(f"fp percentage = {frac_metrics['fpp']}")
print(f"tn percentage = {frac_metrics['tnp']}")
print(f"fn percentage = {frac_metrics['fnp']}")
print(" - - - - - - - - - - - - ")

plot_missclassifications(frac_m0, frac_m1)
plot_tp_and_tn(frac_tp, frac_tn, nofrac_data)
#################################################################

writer = pd.ExcelWriter(test_output_dir + f"{config_dict['model_name']}" + f"_test_metrics.xlsx", engine='xlsxwriter')

df_test_nofractures = pd.DataFrame.from_dict(nofrac_metrics)
df_test_fractures = pd.DataFrame.from_dict(frac_metrics)
df_test_nofractures.to_excel(writer, sheet_name="nofractures")
df_test_fractures.to_excel(writer, sheet_name="fractures")

writer.close()
