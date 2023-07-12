import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import cv2
class OsteopeniaDataset(Dataset):

        def __init__(self, data_file_path, desired_image_size: int):
            self.data_file = pd.read_csv(data_file_path)
            self.desired_image_size = desired_image_size

        def __len__(self):
            return len(self.data_file)

        def __getitem__(self, index):
            image = self._get_image_by_index(index)
            label = self._get_label_by_index(index)
            image = np.moveaxis(image, -1, 0)
            image = image/255.
            return image, label

        def _get_image_by_index(self, index):
            path = self.data_file.loc[index, 'filestem']
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return self._pad_image(img)

        def _get_label_by_index(self, index):
            lbl = self.data_file.loc[index, 'osteopenia']
            if self._is_NaN(lbl):
                return 0
            return lbl

        def _is_NaN(self, num):
            return num != num

        def _pad_image(self, image):
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
            # Grab the old_size
            _old_size = image.shape[:2]  # old_size is in (height, width) format
            # Calculate new size
            _ratio = float(image_desired_size) / max(_old_size)
            _new_size = tuple([int(_x * _ratio) for _x in _old_size])

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

    ds = OsteopeniaDataset(
        '/home/mateo/fakultet/research/osteopenia/osteopenia_dataset/osteopenia_dataset.csv',
        224
    )

    im, lbl = ds[255]
    print(im[0,155])
    print(len(ds))
    plt.figure()
    plt.imshow(im)
    plt.show()
    print(type(im))
    print(im.shape)
    print(lbl)