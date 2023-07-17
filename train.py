from train.trainer import Trainer
from torchvision.models import vgg11, VGG11_Weights, resnet50, resnet34, resnet18, ResNet50_Weights, ResNet18_Weights, \
    ResNet34_Weights
from model.ResNet import ResNet
from model.VGG import VGG
from torch import nn

import torch
import json
import os


def test():
    m = choose_model(model_name, False)
    o = torch.optim.AdamW(m.parameters(), lr=config_dict["learning_rate"], weight_decay=1)

    m, o, m_e = load_model(os.path.join(os.getcwd(), f"model/trained/best-{model_name}.pt"), m,
                           o)

    t = Trainer(
        config_file_path,
        m,
        model_name,
        o,
        nn.BCELoss(),
        m_e
    )

    test_acc, test_loss = t.test_model()
    print(f"Test Loss: {test_loss}, Test accuracy: {test_acc}")


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


######################################################################

config_file_path = os.path.join(os.getcwd(), 'config.json')

with open(config_file_path, "r") as config_file:
    config_dict = json.load(config_file)

model_name = config_dict['model_name']

trainer = None
if config_dict["train_head_first"]:
    model = choose_model(model_name, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_dict["learning_rate"], weight_decay=1)

    trainer = Trainer(
        config_file_path,
        model,
        model_name,
        optimizer,
        nn.BCELoss(),
        1
    )

    trainer.train()

    if config_dict["test_model"]:
        test()

model = choose_model(model_name, False)
optimizer = torch.optim.AdamW(model.parameters(), lr=config_dict["learning_rate"], weight_decay=1)

model, optimizer, model_epoch = load_model(os.path.join(os.getcwd(), f"model/trained/best-{model_name}.pt"), model,
                                           optimizer)

if trainer is None:
    trainer = Trainer(
        config_file_path,
        model,
        model_name,
        optimizer,
        nn.BCELoss(),
        model_epoch
    )
else:
    trainer.model = model
    trainer.optimizer = optimizer

trainer.train()

if config_dict["test_model"]:
    test()
