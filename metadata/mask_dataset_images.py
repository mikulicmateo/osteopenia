import os
import pandas as pd
import json
import cv2
import numpy as np

PROJECT_PATH = os.path.dirname(os.getcwd())
FILTERED_DATASET_DIRECTORY_PATH = os.path.join(PROJECT_PATH, "filtered_dataset/dataset")
CSV_PATH = os.path.join(FILTERED_DATASET_DIRECTORY_PATH, 'filtered_dataset.csv')
data = pd.read_csv(CSV_PATH)
CLASS_TITLES = ["fracture", "metal", "periostealreaction"]


for i in range(len(data)):
    image_path = data.iloc[i]["filestem"]
    with open(f"{image_path[:-4]}.json") as json_file:
        json_dict = json.load(json_file)

    for title in CLASS_TITLES:
        if json_dict[title]:
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = np.ones_like(original_image) * 255
            for point_set in json_dict[title]:
                point_set = np.array(point_set, dtype=np.int32)
                if title == "periostealreaction":
                    cv2.fillPoly(image, pts=[point_set], color=(0, 0, 0))
                else:
                    image[point_set[0][1]:point_set[1][1], point_set[0][0]:point_set[1][0]] = 0
            path_split = image_path.split("/")[:-1]
            save_path = ""
            for word in path_split:
                save_path += f"{word}/"
            
            masked_image = cv2.bitwise_and(original_image, image)
            cv2.imwrite(os.path.join(save_path, f"{title}_removal_mask.png"), masked_image)
