from train import train
import json
import os


model_names = ["resnet18", "resnet34", "resnet50", "vgg11", "vgg16", "vgg19", "efficientnet_b3",
               "densenet121", "densenet169", "densenet201", "mobilenetv3_small", "mobilenetv3_large"]

config_file_path = os.path.join(os.getcwd(), 'config.json')
with open(config_file_path, "r") as config_file:
    config_dict = json.load(config_file)

for model_name in model_names:
    config_dict["model_name"] = model_name
    train(config_dict)