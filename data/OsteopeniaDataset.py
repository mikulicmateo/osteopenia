import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import json
import imutils
import math


class OsteopeniaDataset(Dataset):

    def __init__(self, data_file_path, mean, std, horizontal_flip_probability, rotation_probability, rotation_angle,
                 brightness, contrast, saturation, hue, desired_image_size: int, annotations_path, remove_fractures=False, center_image_by_axis=True):
        self.data_file = pd.read_csv(data_file_path)
        self.desired_image_size = desired_image_size
        self.rotation_probability = rotation_probability
        self.rotation_angle = rotation_angle
        self.annotations_path = annotations_path
        self.remove_fractures = remove_fractures
        self.center_image_by_axis = center_image_by_axis

        if brightness == 0:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomHorizontalFlip(horizontal_flip_probability),
                torchvision.transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomHorizontalFlip(horizontal_flip_probability),
                torchvision.transforms.ColorJitter(brightness=brightness,
                                                   contrast=contrast,
                                                   saturation=saturation,
                                                   hue=hue),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ])

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, index):
        image = self._get_image_by_index(index)
        label = self._get_label_by_index(index)
        # image = np.moveaxis(image, -1, 0)
        image = self.transform(image)
        return image, label

    def _get_image_by_index(self, index):
        path = self.data_file.loc[index, 'filestem']
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        name = self._get_name_by_index(index)
        if self.remove_fractures:
            img = self._remove_fracture(self._get_name_by_index(index), img)
        if not self.center_image_by_axis:
            name = None
        return self._pad_image(img, name)

    def _get_label_by_index(self, index):
        lbl = self.data_file.loc[index, 'osteopenia']
        if self._is_NaN(lbl):
            return 0
        return lbl

    def _get_name_by_index(self, index):
        path = self.data_file.loc[index, 'filestem']
        name = path.split("/")[-1].split(".")[0]
        return name

    def _is_NaN(self, num):
        return num != num

    def _remove_fracture(self, name, img):
        with open(os.path.join(self.annotations_path, f"{name}.json")) as file:
            data = json.load(file)

        for object in data["objects"]:
            title = object.get("classTitle")
            if title == "fracture":
                points = object["points"]["exterior"]
                img[int(points[0][1]): int(points[1][1]), int(points[0][0]): int(points[1][0])] = 0

        return img

    def rotation(self, img, angle):
        angle = int(np.random.uniform(-angle, angle))
        size_reverse = np.array(img.shape[1::-1])  # swap x with y
        M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
        MM = np.absolute(M[:, :2])
        size_new = MM @ size_reverse
        M[:, -1] += (size_new - size_reverse) / 2.
        return cv2.warpAffine(img, M, tuple(size_new.astype(int)))

    def get_axis_points(self, name):
        with open(os.path.join(self.annotations_path, f"{name}.json")) as file:
            data = json.load(file)

        for object in data["objects"]:
            title = object.get("classTitle")
            if title == "axis":
                points = object["points"]["exterior"]
                break

        P1 = points[1]
        P2 = points[0]

        return np.array(P1), np.array(P2)
    def center_image_axis(self, name, image):  # upper point is P1, lower P2
        P1, P2 = self.get_axis_points(name)

        if (P2[1] - P1[1]) < 0:
            y_axis_vector = np.array([0, -1])
        else:
            y_axis_vector = np.array([0, 1])

        if (P2[0] - P1[0]) < 0 and (P2[1] - P1[1]):
            y_axis_vector = np.array([0, 1])

        p_unit_vector = (P2 - P1) / np.linalg.norm(P2 - P1)
        angle_p_y = np.arccos(np.dot(p_unit_vector, y_axis_vector)) * 180 / math.pi

        return imutils.rotate_bound(image, angle_p_y)

    def _pad_image(self, image, name=None):
        """
        Script for resizing the image to the desired dimensions
        First the image is resized then it is zero-padded to the desired
        size given in the argument

        Args:
            * image_desired_dimension, int, new size of the image

        Output:
            * None, self.image is updated
        """
        image_desired_size = self.desired_image_size

        # align axis
        if name is not None:
            image = self.center_image_axis(name, image)

        # Grab the old_size
        _old_size = image.shape[:2]  # old_size is in (height, width) format
        # Calculate new size
        _ratio = float(image_desired_size) / max(_old_size)
        _new_size = tuple([int(_x * _ratio) for _x in _old_size])

        if np.random.rand() < self.rotation_probability:
            image = self.rotation(image, self.rotation_angle)

        # new_size should be in (width, height) format
        image = cv2.resize(image, (_new_size[1], _new_size[0]))

        # Calculate padding
        _delta_w = image_desired_size - _new_size[1]
        _delta_h = image_desired_size - _new_size[0]
        _top, _bottom = _delta_h // 2, _delta_h - (_delta_h // 2)
        _left, _right = _delta_w // 2, _delta_w - (_delta_w // 2)

        # Pad
        color = [0, 0, 0]

        image = cv2.copyMakeBorder(image, _top, _bottom, _left, _right, cv2.BORDER_CONSTANT, value=color)
        # Change to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image


if __name__ == '__main__':

    config_file_path = "/home/mateo/fakultet/research/osteopenia/config.json"

    with open(config_file_path, "r") as config_file:
        config_dict = json.load(config_file)


    ds = OsteopeniaDataset(
        "/home/mateo/fakultet/research/osteopenia/data/test_dataset.csv",
        config_dict['mean'],
        config_dict['std'],
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        config_dict['desired_image_size'],
        config_dict['additional_annotations_path']
    )

    im, lbl = ds[0]
    print(im)
    print(im[0, 155])
    print(len(ds))
    print(im.shape)
    #plt.figure()
    #im = np.moveaxis(im, 0, -1)
    #print(type(im))
    #print(im.shape)
    #cv2.imshow("bo", im.numpy())
    indices = np.random.randint(0, len(ds), size=2)

    df = pd.read_csv("/home/mateo/fakultet/research/osteopenia/data/test_dataset.csv")
    paths = df['filestem'].to_numpy()

    for i in indices:
        im, lbl = ds[i]
        f, axarr = plt.subplots(1, 2, figsize=(15, 15))
        image = plt.imread(paths[i])
        axarr[0].imshow(im.permute(1, 2, 0), cmap='gray')
        axarr[1].imshow(image, cmap='gray')
        axarr[0].axis('off')
        axarr[1].axis('off')
        for ax in f.axes:
            ax.axison = False
        plt.savefig(f"side_by_side_{i}.png")
    #plt.imshow(im[0])
    #plt.show()
    print(lbl)