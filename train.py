from train.trainer import Trainer
from torchvision.models import resnet34, ResNet34_Weights
from model.ResNet import ResNet
from torch import nn

import torch
import json
import os

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

model_name = 'resnet34'
trainer = None

if config_dict["train_head_first"]:
        model = ResNet(resnet34(weights=ResNet34_Weights.DEFAULT), freeze=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config_dict["learning_rate"], weight_decay=1e-05)

        trainer = Trainer(
            config_file_path,
            model,
            'resnet34',
            optimizer,
            nn.BCELoss(),
            1
        )

        trainer.train()

        # if config_dict["test_model"]:
        #     trainer.test_model()

model = ResNet(resnet34(weights=ResNet34_Weights.DEFAULT), freeze=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=config_dict["learning_rate"], weight_decay=1e-05)

model, optimizer, model_epoch = load_model(os.path.join(os.getcwd(), f"model/trained/best-{model_name}.pt"), model, optimizer)

if trainer is None:
    trainer = Trainer(
            config_file_path,
            model,
            'resnet34',
            optimizer,
            nn.BCELoss(),
            model_epoch
    )
else:
    trainer.model = model
    trainer.optimizer = optimizer

    # if config_dict["test_model"]:
    #     trainer.test_model()

trainer.train()

