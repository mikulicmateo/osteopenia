import os
import pandas as pd
import json
import shutil

PROJECT_PATH = os.getcwd()
DATASET_DIRECTORY_PATH = os.path.join(PROJECT_PATH, "dataset")
CSV_PATH = os.path.join(DATASET_DIRECTORY_PATH, "dataset.csv")

config_file_path = os.path.join(os.getcwd(), 'config.json')
with open(config_file_path, "r") as config_file:
    config_dict = json.load(config_file)

OSTEOPENIA_DATASET_DIRECTORY_PATH = os.path.join(PROJECT_PATH, "osteopenia_dataset")

data = pd.read_csv(CSV_PATH)

#print(sum(data['osteopenia'] == 1))

#extract patients that have osteopenia at some point
k = set()
for i in range(len(data)):
    if data.iloc[i]['osteopenia'] == 1:
        k.add(data.iloc[i]['patient_id'])


#dicts of patients that have osteopenia for each image
dict_list = []
for key in k:
    df = data.loc[data['patient_id'] == key]
    folder_path = os.path.join(OSTEOPENIA_DATASET_DIRECTORY_PATH, str(key))
    for j in range(len(df)):
        filter_out = False
        dict = df.iloc[j].to_dict()
        if dict["cast"] == 1:
            filter_out = True
        image_name = dict['filestem']
        with open(os.path.join(config_dict["additional_annotations_path"], f"{image_name}.json")) as file:
            json_file = json.load(file)
            for obj in json_file["objects"]:
                title = obj.get("classTitle")
                if title == "metal": #or title == "periostealreaction":
                    filter_out = True

        if filter_out:
            continue

        dict['filestem'] = os.path.join(folder_path, dict['filestem'] + '.png')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        image_src_path = os.path.join(DATASET_DIRECTORY_PATH, "images", image_name + ".png")

        dst_path = os.path.join(folder_path, image_name)
        shutil.copy2(image_src_path, dst_path + ".png")

        dict_list.append(dict)
        json_object = json.dumps(dict, indent=4)
        with open(dst_path + ".json", "w") as outfile:
            outfile.write(json_object)

csv_df = pd.DataFrame.from_records(dict_list)
csv_df.to_csv(os.path.join(OSTEOPENIA_DATASET_DIRECTORY_PATH, "osteopenia_dataset.csv"), encoding='utf-8', index=False)
