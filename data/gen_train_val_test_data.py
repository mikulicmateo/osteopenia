import json
import pandas as pd
import os

from sklearn.model_selection import train_test_split

config_file_path = os.path.join(os.path.dirname(os.getcwd()), 'config.json')

with open(config_file_path, "r") as config_file:
    config_dict = json.load(config_file)

random_state = config_dict["random_state"]
full_dataset = pd.read_csv(config_dict['osteopenia_dataset_csv_path']).sample(frac=1, random_state=random_state) #shuffled dataset

images = full_dataset['filestem'].to_numpy()
labels = full_dataset['osteopenia'].fillna(0).to_numpy()

train_percentage = config_dict['n_split_train_val_test'][0]
val_percentage = config_dict['n_split_train_val_test'][1]
test_percentage = config_dict['n_split_train_val_test'][2]

x_train, X_test, y_train, Y_test = train_test_split(images, labels, test_size=1 - train_percentage, random_state=random_state) # split to get training

x_val, x_test, y_val, y_test = train_test_split(X_test, Y_test, test_size=test_percentage/(1 - train_percentage), random_state=random_state) # split to get val & test

# create dfs
training_df = pd.DataFrame({'filestem':x_train, 'osteopenia':y_train})
validation_df = pd.DataFrame({'filestem':x_val, 'osteopenia':y_val})
test_df = pd.DataFrame({'filestem':x_test, 'osteopenia':y_test})

# save
training_df.to_csv(os.path.join(os.getcwd(), 'training_dataset.csv'))
validation_df.to_csv(os.path.join(os.getcwd(), 'validation_dataset.csv'))
test_df.to_csv(os.path.join(os.getcwd(), 'test_dataset.csv'))
