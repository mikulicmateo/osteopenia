import json
import os
from util.train_util import *
from train.trainer import Trainer
from torch import nn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


config_file_path = os.path.join(os.getcwd(), 'config.json')

with open(config_file_path, "r") as config_file:
    config_dict = json.load(config_file)

model_name = config_dict['model_name']

model = choose_model(model_name, False)
optimizer = torch.optim.AdamW(model.parameters())

model, optimizer, epoch = load_model(os.path.join(os.getcwd(), 
                                                  f"model/trained/best-{model_name}.pt"),
                                                  model,
                                                  optimizer,
                                                  config_dict['device'],
                                                  config_dict["device_ids"])

trainer = Trainer(
            config_dict,
            model,
            model_name,
            optimizer,
            nn.BCELoss(),
            epoch,
            "model/results/",
            True
        )

trainer.test_dl = trainer.create_dataloader(trainer.test_data, 1, 4, False)

loop = tqdm(trainer.test_dl, leave=False, unit="batch", mininterval=0)

tp = 0
fp = 0
fn = 0
tn = 0
indices_m0 = []
indices_m1 = []

k = 0
for image, label in loop:
    prediction = trainer.predict(image).detach().cpu().numpy()[0,0]
    if prediction >= 0.5:
        if label == 1:
            tp += 1
        else:
            fp += 1
            indices_m0.append(k)
    else:
        if label == 1:
            fn += 1
            indices_m1.append(k)
        else:
            tn += 1
    k += 1

precision = tp/(tp + fp)
recall = tp/(tp + fn)

f1 = 2 * (precision * recall) / (precision + recall)

print(" - - - - SCORES - - - - ")
print(f"F1-score = {f1}")
print(f"precision = {precision}, recall = {recall}")
print("- - - - - - - - - - - - ")
print(f"tp percentage = {tp/(tp+fp+fn+tn)}")
print(f"fp percentage = {fp/(tp+fp+fn+tn)}")
print(f"tn percentage = {tn/(tp+fp+fn+tn)}")
print(f"fn percentage = {fn/(tp+fp+fn+tn)}")
print("- - - - - - - - - - - - ")

df = pd.read_csv(os.path.join(os.getcwd(), "data", "test_dataset.csv"))
paths = df['filestem'].to_numpy()
misclassified_0 = [paths[k] for k in indices_m0]
misclassified_1 = [paths[k] for k in indices_m1]


f, axarr = plt.subplots((len(misclassified_0)//2), 2, figsize=(15,15))
f.suptitle("False Positives", fontsize=30)

row = 0
column = 0
for i in range(len(misclassified_0)):
    image = plt.imread(misclassified_0[i])
    axarr[row, column].imshow(image, cmap='gray')
    axarr[row, column].axis('off')
    if i%2 == 0:
        if column == 1:
            column = 0
        else:
            column = 1
    else:
        row += 1

for ax in f.axes:
    ax.axison = False
plt.savefig("missclasified_0.png")
plt.show()

f, axarr = plt.subplots((len(misclassified_1)//2)+1, 2, figsize=(20,20))
f.suptitle("False Negatives", fontsize=30)

row = 0
column = 0
for i in range(len(misclassified_1)):
    image = plt.imread(misclassified_1[i])
    axarr[row, column].imshow(image, cmap='gray')
    if i%2 == 0:
        if column == 1:
            column = 0
        else:
            column = 1
    else:
        row += 1

for ax in f.axes:
    ax.axison = False
plt.savefig("missclasified_1.png")
plt.show()
