from train import train, diffusion_train
import json
import os


model_names = ["resnet18", "resnet34", "resnet50", "vgg11", "vgg16", "vgg19", "efficientnet_b3",
               "densenet121", "densenet169", "densenet201", "mobilenetv3_small", "mobilenetv3_large"]

config_file_path = os.path.join(os.getcwd(), 'config.json')
with open(config_file_path, "r") as config_file:
    config_dict = json.load(config_file)

# for model_name in model_names:
#     config_dict["model_name"] = model_name

#     if config_dict["diffusion_train"]:
#         diffusion_train(config_dict)
#     else:
#         train(config_dict)

fracture_mask_probabilities = [0, 0.33]
metal_mask_probabilities = [0, 0.5]
periosteal_mask_probabilities = [0, 0.255]

for model_name in model_names:
    for j in range(2):
        for i in range(0, 10):
            config_dict['fracture_mask_probability'] = fracture_mask_probabilities[j]
            config_dict['metal_mask_probability'] = metal_mask_probabilities[j]
            config_dict['periosteal_mask_probability'] = periosteal_mask_probabilities[j]
            config_dict['model_id'] = i
            config_dict['results_output_dir'] = f"model/{model_name}_results/{config_dict['model_id']}_fm{config_dict['fracture_mask_probability']}_mm{config_dict['metal_mask_probability']}_pm{config_dict['periosteal_mask_probability']}/"
            
            config_dict["model_name"] = model_name
            
            if config_dict["diffusion_train"]:
                diffusion_train(config_dict)
            else:
                train(config_dict)