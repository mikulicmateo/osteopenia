import os
import pandas as pd
import json
import shutil

PROJECT_PATH = os.path.dirname(os.getcwd())
DATASET_DIRECTORY_PATH = os.path.join(PROJECT_PATH, "dataset")
ANNOTATIONS_DIRECTORY_PATH = os.path.join(DATASET_DIRECTORY_PATH, "annotations_all")
CSV_PATH = os.path.join(DATASET_DIRECTORY_PATH, "dataset.csv")
FILTERED_DATASET_DIRECTORY_PATH = os.path.join(PROJECT_PATH, "filtered_dataset/dataset")

data = pd.read_csv(CSV_PATH)
# Drop the metal column to later replace it with points for creating masks
data = data.drop(columns=["metal"])

# Extract patients that have osteopenia at some point
k = set()
for i in range(len(data)):
    if data.iloc[i]['osteopenia'] == 1:
        k.add(data.iloc[i]['patient_id'])

CLASS_TITLES = ["fracture", "metal", "periostealreaction"]

# Dicts of patients that have osteopenia for each image
dict_list = []
for key in k:
    df = data.loc[data['patient_id'] == key]
    patient_path = os.path.join(FILTERED_DATASET_DIRECTORY_PATH, f"patient_id_{str(key)}")
    for j in range(len(df)):
        folder_path = os.path.join(patient_path, f"image_{str(j+1)}")
        row_dict = df.iloc[j].to_dict()
        image_name = row_dict['filestem']

        with open(os.path.join(ANNOTATIONS_DIRECTORY_PATH, f"{image_name}.json")) as annotations_file:
            annotations_dict = json.load(annotations_file)

        # Filter dataset
        filter_out = False
        for object in annotations_dict["tags"]:
            if object == "cast":
                filter_out = True
                break

        if filter_out:
            continue

        # Add class title points to json
        for title in CLASS_TITLES:
                row_dict[title] = []

        for object in annotations_dict["objects"]:
            title = object.get("classTitle")
            if title in CLASS_TITLES:
                points = object["points"]["exterior"]
                row_dict[title].append(points)

        # Add image path to dict
        row_dict['filestem'] = os.path.join(folder_path, row_dict['filestem'] + '.png')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        image_src_path = os.path.join(DATASET_DIRECTORY_PATH, "images", image_name + ".png")

        # Copy image to new location
        dst_path = os.path.join(folder_path, image_name)
        shutil.copy2(image_src_path, dst_path + ".png")

        # Write json object
        dict_list.append(row_dict)
        json_object = json.dumps(row_dict, indent=4)
        with open(dst_path + ".json", "w") as outfile:
            outfile.write(json_object)

# Save CSV
csv_df = pd.DataFrame.from_records(dict_list)
csv_df.to_csv(os.path.join(FILTERED_DATASET_DIRECTORY_PATH, "filtered_dataset.csv"), encoding='utf-8', index=False)
