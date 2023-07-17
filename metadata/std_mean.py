import os.path
import numpy as np
from data.OsteopeniaDataset import OsteopeniaDataset
from torch.utils.data import DataLoader
import json
import os
import torch

with open(os.path.join(os.path.dirname(os.getcwd()), 'config.json'), "r") as config_file:
    config_dict = json.load(config_file)

ds = OsteopeniaDataset(
    os.path.join(os.path.dirname(os.getcwd()), 'data/training_dataset.csv'),
    0,
    1,
    config_dict['desired_image_size']
)

pix_sum = torch.tensor([0.0, 0.0, 0.0])
pix_sum_sq = torch.tensor([0.0, 0.0, 0.0])

dl = DataLoader(ds, batch_size=32, num_workers=4)

for image, label in dl:
    pix_sum += image.sum(axis=[0, 2, 3])
    pix_sum_sq += (image**2).sum(axis=[0, 2, 3])

pix_count = len(ds) * config_dict['desired_image_size'] * config_dict['desired_image_size']

total_mean = pix_sum / pix_count
total_var = (pix_sum_sq / pix_count) - (total_mean ** 2)
total_std = torch.sqrt(total_var)

print('mean: ' + str(total_mean))
print('std:  ' + str(total_std))


total = 0.0
totalsq = 0.0
count = 0

for data, *_ in dl:
    count += np.prod(data.shape)
    total += data.sum()
    totalsq += (data**2).sum()

mean = total/count
var = (totalsq/count) - (mean**2)
std = torch.sqrt(var)

print(mean)
print(var)
print(std)
