from trainer.trainer import Trainer
from util.train_util import *
import torch
import json
import os
import numpy as np


def test(config_dict, stage=-1):
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

    t = Trainer(
        config_dict,
        m,
        config_dict["model_name"],
        o,
        nn.BCELoss(),
        m_e,
        test_output_dir,
        True
    )

    test_acc, test_loss = t.test_model()
    print(f"Test Loss: {test_loss}, Test accuracy: {test_acc}")

    test_dict = {}
    test_dict['test_acc'] = test_acc.cpu().detach().item()
    test_dict['test_loss'] = test_loss

    if stage == -1:
        file_name = f"{config_dict['model_name']}_lr{config_dict['learning_rate']}_epoch{m_e}.json"
    else:
        file_name = f"{config_dict['model_name']}_lr{config_dict['learning_rate']}_stage{stage}_epoch{m_e}.json"

    test_output = test_output_dir + file_name
    json_object = json.dumps(test_dict, indent=4)
    with open(test_output,"w") as outfile:
        outfile.write(json_object)


######################################################################
def diffusion_start_load(model_name, config_dict):
    model = choose_model(model_name, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_dict["learning_rate"])

    trainer = Trainer(
        config_dict,
        model,
        model_name,
        optimizer,
        nn.BCELoss(),
        1,
        "model/results/"
    )

    return trainer


def diffusion_reset_load(model_name, path, learning_rate, config_dict):
    model = choose_model(model_name, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model, _, model_epoch = load_model(os.path.join(os.getcwd(), path),
                                       model,
                                       None,
                                       config_dict["device"],
                                       config_dict["device_ids"])

    trainer = Trainer(
        config_dict,
        model,
        model_name,
        optimizer,
        nn.BCELoss(),
        model_epoch,
        "model/results/",
        True
    )

    return trainer


def train(config_dict):
    model_name = config_dict['model_name']

    trainer = None
    if config_dict["train_head_first"]:
        model = choose_model(model_name, True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config_dict["learning_rate"], weight_decay=1)

        trainer = Trainer(
            config_dict,
            model,
            model_name,
            optimizer,
            nn.BCELoss(),
            1,
            "model/results/"
        )

        trainer.train()

        if config_dict["test_model"]:
            test(config_dict)

    model = choose_model(model_name, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_dict["learning_rate"], weight_decay=1)

    model, _, model_epoch = load_model(os.path.join(os.getcwd(),
                                                    f"model/trained/best-{model_name}.pt"),
                                       model,
                                       None,
                                       config_dict["device"],
                                       config_dict["device_ids"])

    if trainer is None:
        trainer = Trainer(
            config_dict,
            model,
            model_name,
            optimizer,
            nn.BCELoss(),
            model_epoch,
            "model/results/",
            True
        )
    else:
        trainer.model = model
        trainer.optimizer = optimizer

    trainer.train()

    if config_dict["test_model"]:
        test(config_dict)


def diffusion_train(config_dict):
    model_name = config_dict['model_name']
    learning_rate = config_dict["learning_rate"]

    if config_dict['freeze_ratio'] is not list:
        freezing_ratios = list(np.arange(1.0, -0.0, -config_dict['freeze_ratio']))
        freezing_ratios.append(0)
        freezing_ratios = freezing_ratios[1:]
    else:
        freezing_ratios = config_dict['freeze_ratio']

    print("########################################################################")
    print("################ Starting process of diffusion training ################")
    print("########################################################################")
    for i, ratio in enumerate(freezing_ratios):
        print(f"******************* Stage {i}/{len(freezing_ratios)-1} *******************")
        print(f"USING MODEL: {model_name}, WEIGHTS: FROZEN, ratio: {ratio}")

        if i == 0:
            trainer = diffusion_start_load(model_name, config_dict)
            trainer.freeze_model_part(ratio)
        else:
            learning_rate = learning_rate * 0.5
            trainer = diffusion_reset_load(model_name, f"model/trained/stage-{i - 1}/best-{model_name}.pt", learning_rate, config_dict)
            trainer.freeze_model_part(ratio)

        trainer.stage = i
        trainer.train()
        print("**************************************")
    print("########################################################################")

    if config_dict["test_model"]:
        for i, _ in enumerate(freezing_ratios):
            print(f"******************* Test best {model_name} stage {i} *******************")
            test(config_dict, i)


if __name__ == "__main__":
    config_file_path = os.path.join(os.getcwd(), 'config.json')

    with open(config_file_path, "r") as config_file:
        config_dict = json.load(config_file)

    if config_dict["diffusion_train"]:
       diffusion_train(config_dict)
    else:
       train(config_dict)
    #test(config_dict, 9)
