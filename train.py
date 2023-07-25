from train.trainer import Trainer
from util.train_util import *
import torch
import json
import os


def test(config_dict):
    m = choose_model(config_dict["model_name"], False)
    o = torch.optim.AdamW(m.parameters(), lr=config_dict["learning_rate"], weight_decay=1)

    m, o, m_e = load_model(os.path.join(os.getcwd(), f"model/trained/best-{config_dict['model_name']}.pt"), m,
                           o, config_dict["device"], config_dict["device_ids"])

    t = Trainer(
        config_dict,
        m,
        config_dict["model_name"],
        o,
        nn.BCELoss(),
        m_e,
        "model/results/",
        True
    )

    test_acc, test_loss = t.test_model()
    print(f"Test Loss: {test_loss}, Test accuracy: {test_acc}")


######################################################################

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


if __name__ == "__main__":
    config_file_path = os.path.join(os.getcwd(), 'config.json')

    with open(config_file_path, "r") as config_file:
        config_dict = json.load(config_file)
    
    train(config_dict)