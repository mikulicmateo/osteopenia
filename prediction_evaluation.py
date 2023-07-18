import json
import os
from torchvision.models import vgg11, VGG11_Weights, resnet50, resnet34, resnet18, ResNet50_Weights, ResNet18_Weights, \
    ResNet34_Weights
from model.ResNet import ResNet
from model.VGG import VGG
from train.trainer import Trainer
from torch import nn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
def choose_model(name, freeze):
    if name == 'vgg11':
        return VGG(vgg11(weights=VGG11_Weights.DEFAULT), freeze=freeze)
    if name == 'resnet34':
        return ResNet(resnet34(weights=ResNet34_Weights.DEFAULT), freeze=freeze)
    if name == 'resnet18':
        return ResNet(resnet34(weights=ResNet34_Weights.DEFAULT), freeze=freeze)


def load_model(model_path, model, optimizer):
    model_dict = torch.load(model_path)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("cuda not available!")
        return None, None, None

    model.load_state_dict(model_dict['model_state'])
    model.to(device)
    model_epoch = model_dict['epoch']

    if optimizer is not None:
        optimizer.load_state_dict(model_dict['optimizer_state'])

    return model, optimizer, model_epoch


config_file_path = os.path.join(os.getcwd(), 'config.json')

with open(config_file_path, "r") as config_file:
    config_dict = json.load(config_file)

model_name = config_dict['model_name']

model = choose_model(model_name, False)
optimizer = torch.optim.AdamW(model.parameters())

model, optimizer, epoch = load_model(os.path.join(os.getcwd(), f"model/trained/best-{model_name}.pt"), model, optimizer)

trainer = Trainer(
            config_file_path,
            model,
            model_name,
            optimizer,
            nn.BCELoss(),
            epoch
        )

trainer.test_dl = trainer.create_dataloader(trainer.test_data, 1, 4, False)

loop = tqdm(trainer.test_dl, leave=False, unit="batch", mininterval=0)

tp = 0
fp = 0
fn = 0
tn = 0

misclassified_0 = []
misclassified_1 = []
for image, label in loop:
    prediction = trainer.predict(image).detach().cpu().numpy()[0,0]
    if prediction >= 0.5:
        if label == 1:
            tp += 1
        else:
            fp += 1
            misclassified_0.append(image.detach().cpu().numpy()[0])
    else:
        if label == 1:
            fn += 1
            misclassified_1.append(image.detach().cpu().numpy()[0])
        else:
            tn += 1


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



f, axarr = plt.subplots((len(misclassified_0)//2), 2, figsize=(15,15))
f.suptitle("False Positives", fontsize=30)

row = 0
column = 0
for i in range(len(misclassified_0)):
    image = np.moveaxis(misclassified_0[i], 0, -1)
    axarr[row, column].imshow(image)
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
plt.show()

f, axarr = plt.subplots((len(misclassified_1)//2)+1, 2, figsize=(20,20))
f.suptitle("False Negatives", fontsize=30)

row = 0
column = 0
for i in range(len(misclassified_1)):
    image = np.moveaxis(misclassified_1[i], 0, -1)
    axarr[row, column].imshow(image)
    if i%2 == 0:
        if column == 1:
            column = 0
        else:
            column = 1
    else:
        row += 1

for ax in f.axes:
    ax.axison = False
plt.show()

