import torch
import torch.cuda
import glob
import os
from collections import namedtuple
import pandas as pd
import json
from collections import *
import random
import diskcache
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2

#**********************************************************************#
# Named tupples for handeling data
"""
Data params: 
root_dir -->path, 
split_ratio --> split ratio for dataset
type--> valid, train, test
image_dimension--> image dimensions
threshold--> data occurances threshold
"""
data_params = namedtuple(
    'data_params',
    'root_dir, split_ratio, type, threshold, image_dimension, verbose',
)


"""
Normalization_params:
px_area_scaler --> normalization for scaling
original_width_scaler --> scaler for width
original_height_scaler --> scaler for hegiht
image_scaler --> dimension of the image
"""
normalization_params = namedtuple(
    'normalization_params',
    'px_area_scaler, original_width_scaler, original_height_scaler, image_scaler',
)


"""
Loader params:
batch_size-->batch size,
number_of_workers --> for loading
dataset_info --> data_params
"""
loader_params = namedtuple(
    'loader_params',
    'batch_size, number_of_workers, use_gpu ,dataset_info',
)



#**********************************************************************#
# Data loading helpful functions

def get_cache(scope_str):
    """
    Cashing Descriptor function
    """
    return diskcache.FanoutCache('data_cache/' + scope_str,
                       shards=20,
                       timeout=1,
                       size_limit=3e11,
                       )

my_cache = get_cache('Cache')

@my_cache.memoize(typed=True)
def get_data_sample(sample, data_normalization_coeficients):
    """
    Middleman function for cashing Fast is smooth, smooth is fast
    """
    _data = ProcessData(sample, data_normalization_coeficients)
    _output = _data.get_sample()
    return _output
#**********************************************************************#
class ProcessData: 
    """
    Class for loading data from json
    """
    def __init__(self, sample, data_normalization_coeficients):
        """
        Init function.

        Args: 
            * sample, dictionary with following attributes: 'org_width', 'org_height', 'side', 'projection', 'type', 
            'original_coordinates', 'px_area', 'relative_area', 'center', 'width', 'Image', 'RootDir'

            * dimensions, integer, scaling factor for images 
        """
        # Obtain extract data

        ## Original width and height
        # Formula: (x - x.min()) / (x.max() - x.min())
        self.org_width = (sample['org_width'] - data_normalization_coeficients.original_width_scaler[0]) / (data_normalization_coeficients.original_width_scaler[1] - data_normalization_coeficients.original_width_scaler[0])

        self.org_height = (sample['org_height'] - data_normalization_coeficients.original_height_scaler[0]) / (data_normalization_coeficients.original_height_scaler[1] - data_normalization_coeficients.original_height_scaler[0])

        ## Area
        self.px_area = (sample['px_area'] - data_normalization_coeficients.px_area_scaler[0]) / (data_normalization_coeficients.px_area_scaler[1] - data_normalization_coeficients.px_area_scaler[0])
        self.relative_px_area = sample['relative_area']

        ## Side
        # Left == 0, right = 1
        if sample['side'] == 'left':
            self.side = 0
        else:
            self.side = 1

        ## Projection
        # lat = 0, ap = 1
        if sample['projection'] == 'lat':
            self.projection = 0
        else:
            self.projection = 1

        ## Type
        # Classes: 
        _class_list = ['23r-M/3.1', '23r-M/2.1', '23u-M/2.1', '23u-E/7', '23u-M/3.1', '23r-E/2.1', '22r-D/2.1', '22u-D/2.1', '22r-D/4.1', '23r-E/1', '22u-D/4.1', '23u-E/2.1', '72B(b)', '22-D/2.1']
        self.label = _class_list.index(sample['type'])

        ## Fracture center width and height 
        self.fract_center_x = sample['center'][0]
        self.fract_center_y = sample['center'][1]
        self.fract_width = sample['width'][0]
        self.fract_height = sample['width'][1]

        ## Cosmetics
        self.original_cordinates = sample['original_coordinates']
        self.img_path = sample['Image']
        self.root_path = sample['RootDir']

        ## Images   
        _image = sample['Image']
        self.image = cv2.imread(_image, cv2.IMREAD_GRAYSCALE)
        self.__pad_image__(data_normalization_coeficients.image_scaler)
        
    def __pad_image__(self, image_desired_size:int):
        """
        Script for resizing the image to the desired dimensions
        First the image is resized then it is zero-padded to the desired 
        size given in the argument

        Args:
            * image_desired_dimension, int, new size of the image

        Output:
            * None, self.image is updated
        """
        # Grab the old_size
        _old_size = self.image.shape[:2] # old_size is in (height, width) format
        # Calculate new size
        _ratio = float(image_desired_size)/max(_old_size)
        _new_size = tuple([int(_x*_ratio) for _x in _old_size])

        # new_size should be in (width, height) format
        self.image = cv2.resize(self.image, (_new_size[1], _new_size[0]))
        
        
        # Calculate padding
        _delta_w = image_desired_size - _new_size[1]
        _delta_h = image_desired_size - _new_size[0]
        _top, _bottom = _delta_h//2, _delta_h-(_delta_h//2)
        _left, _right = _delta_w//2, _delta_w-(_delta_w//2)
        
        # Pad
        color = [0, 0, 0]
        
        self.image = cv2.copyMakeBorder(self.image, _top, _bottom, _left, _right, cv2.BORDER_CONSTANT, value=color)
        # Change to grayscale
        
        self.image = self.image
        
    def get_sample(self):
        """
        Return sample --> loaded image and its annotations
        """
        return (self.image, self.label, self.org_height, self.org_width, 
                self.fract_center_x, self.fract_center_y, self.fract_height, self.fract_width, 
                self.px_area, self.relative_px_area, self.projection, self.side,
                self.root_path, self.img_path, self.original_cordinates)

#**********************************************************************#

# Main class for handeling dataset
class img_retreival_dataset:
    """
    Class for handling the dataset. It is the plain train,valid,split

    Random seed is set to 1221
    """

    def __init__(self, dataset_info:data_params):
        """
        Init function which handles the dataset loading
        Args:
            * Dataset_info, data_params (named_tupple):
                --> root_dir, string, path to directory
                --> split_ratiom, float, how many percent is for 
                                    train dataset,
                --> type, string, train, valid, test
                --> threshold, int, data occurances threshold
                --> image_dimension, tupple, image dimensions
                --> verbose, bool, for debuging
        """

        # Set random seed
        random.seed(1221)
        
        # Check if data path is ok
        assert os.path.exists(dataset_info.root_dir), f"Path {dataset_info.root_dir} does not exist"

        # Obtain data lists --> all data as json
        _data_list = self.obtain_structured_data(dataset_info.root_dir)
        
        # Filter out data based on number of diagnoses occurances
        _filtered_data_list = self.filter_out_data(_data_list, 
                                                   dataset_info.threshold)

        # Create splits
        _data_list_splitted = self.split_to_train_val_test(_filtered_data_list, 
                                                                            split_ratio = dataset_info.split_ratio, 
                                                                            verbose = dataset_info.verbose)   

        # Select dataset and store it in self.data 
        selector = lambda type, data_list_splitted: {
        'train': data_list_splitted[0],
        'valid': data_list_splitted[1],
        'test': data_list_splitted[2],
        }[type]        
        self.data_list = selector(dataset_info.type, _data_list_splitted)

        # Obtain normalization stats on training set
        self.data_normalization_coeficients = self.calculate_normalization_coefficients(_data_list_splitted[0], 
                                                                                        dataset_info.image_dimension)
        # Get count it
        self.samples_cnt = len(self.data_list)

    def calculate_normalization_coefficients(self, input_list:list, image_dimension: int)->normalization_params:
        """
        Function which calculates min and max values for the features that needs normalization

        Args:
            * input_list, list, list which contains training data samples 
            * image_dimension, int, number representing the size of the input image to be rescaled

        Output:
            * normalization_params, named tupple with normalization params
        """
        # Create storing unit for data
        _storage = {'px_area_scaler': [], 'original_width_scaler': [], 
                  'original_height_scaler': []}
        
        # Go trough data 
        for _data_sample in input_list:
            _storage['original_width_scaler'].append(_data_sample['org_width'])
            _storage['original_height_scaler'].append(_data_sample['org_height'])
            _storage['px_area_scaler'].append(_data_sample['px_area'])

        # Calculate values    
        _px_area_scaler = (np.min(_storage['px_area_scaler']), np.max(_storage['px_area_scaler']))
        _original_height_scaler = (np.min(_storage['original_height_scaler']), np.max(_storage['original_height_scaler']))
        _original_width_scaler = (np.min(_storage['original_width_scaler']), np.max(_storage['original_width_scaler']))
        
        # Return normalization params
        return normalization_params(px_area_scaler = _px_area_scaler,
                                              original_height_scaler = _original_height_scaler,
                                              original_width_scaler = _original_width_scaler,
                                              image_scaler = image_dimension)
    
    def __len__(self):
        """
        Returns number of samples in dataset
        """
        return self.samples_cnt
    
    def shuffle_samples(self):
        """
        Simply shuffles the dataset -- necessary for batches
        """
        # Shuffeling dataset
        random.seed(1221)
        random.shuffle(self.data_list)

    def __getitem__(self, indx):
        """
        Gets data from the dataset

        Args:
            * indx, int, index of the data sample
        
        Output: data sample
        """
        # Get sample
        _sample = self.data_list[indx]

        # Obtain image (input) and annotation(output)
        _preprocesed_data = get_data_sample(_sample, self.data_normalization_coeficients)     

        # Image
        _image = torch.from_numpy(_preprocesed_data[0])
        _image = _image.to(torch.float32)
        _image /= 255.0
        
        # label
        _label = torch.tensor(_preprocesed_data[1])

        # original witdht, original height
        _org_width = torch.tensor(_preprocesed_data[2])
        _org_height = torch.tensor(_preprocesed_data[3])

        # fracture characteristics
        _fract_center_x = torch.tensor(_preprocesed_data[4])
        _fract_center_y = torch.tensor(_preprocesed_data[5])
        _fract_height = torch.tensor(_preprocesed_data[6])
        _fract_width = torch.tensor(_preprocesed_data[7])
        _fract_px_area = torch.tensor(_preprocesed_data[8])
        _fract_relative_px_area = torch.tensor(_preprocesed_data[9])
            
        # image description
        _projection = torch.tensor(_preprocesed_data[10])
        _side = torch.tensor(_preprocesed_data[11])
        
        # case description
        _root_path = _preprocesed_data[12]
        _img_path = _preprocesed_data[13]
        _original_coordinates = _preprocesed_data[14]

        # Return
        return (_image, _label, _org_width, _org_height, 
                _fract_center_x, _fract_center_y, _fract_height, _fract_width,
                _fract_px_area, _fract_relative_px_area,
                _projection, _side, _root_path, _img_path, _original_coordinates)

    def obtain_structured_data(self, path:str)->dict:
        """
        Function which list files and builds up dict with all relevent data
        
        Args:
            * Path to files 
        Output:
            * dict where keys are diagnoses types, and values are list dictornaries related to the fractures.
            The fractures are metadata with addition of _fracture_data type, and path to the
            root dir and image.
        
        """
        
        # Save dir
        _main_output_dir = {}
        
        # Obtain file names
        _dir_list = os.listdir(path)
        _add_path = lambda _file_names, _path, _target: [os.path.join(_path, _f, _target) for _f in _file_names]
        _file_list = _add_path(_dir_list, path, 'metadata.json')
        _image_list = _add_path(_dir_list, path, 'Fracture_images')
        
        # Go trough files and export data
        for _file_name, _image_name in zip(_file_list, _image_list):
            
            # Load data
            with open(_file_name) as _file:
                _data = json.load(_file)
            
            # Load images
            _images_paths = []
            for _i in range(0, len(os.listdir(_image_name))):
                _images_paths.append(os.path.join(_image_name, f"{_i}.png"))
            
            # Merge
            for _i, _key in enumerate(_data):
                # Obtain data
                _fracture_data = _data[_key]
                _fracture_data['Image'] = _images_paths[_i]
                _fracture_data['type'] = _key
                _fracture_data['RootDir'] = os.path.dirname(_file_name)
                
                # Save data into dir
                if _key in _main_output_dir:
                    _main_output_dir[_key].append(_fracture_data)
                else:
                    _main_output_dir[_key] = []
                    _main_output_dir[_key].append(_fracture_data)
                    
        # Return data
        return _main_output_dir

    def filter_out_data(self, input_dict:dict, minumum_frequency:int = 20)->dict:
        """
        Function which filter out only fractures with 20 or more occurances
        
        Args:
            * Dictionary obtained by the function: obtain_structured_data
            * minimum frequency: Minimum number of occurances of file
        
        Output: filtered dict
        """
        # Sort dictionary
        _input_dict = OrderedDict(sorted(input_dict.items(), key=lambda x: len(x[1]), reverse=True))
        
        # Create output dir
        _main_output_dir = _input_dict.copy()
        
        # Filter
        for _item in _input_dict:
            if len(_input_dict[_item]) < minumum_frequency:
                del _main_output_dir[_item] 
        
        # Export
        return _main_output_dir    

    def split_to_train_val_test(self, data_dict:dict, split_ratio :float = 0.75, verbose : bool = 0)->list:
        """
        Function which accepts parsed dict by filtered out data and obtained structured path
        
        Args:
            * data_dict, dictionary, dictionary which containes out of fractures
            * train_split_ratio, float, split ratio for dataset. valid=test=1-_train_split_ratio/2
            * verbose, boolean, print statistics on the run
        
        Output:
            * three lists of samples: train, valid, test.
        """
        # Set random seed
        random.seed(1221)
        
        # Set split ratio # Valid/test is what is left from
        _train_split_ratio = split_ratio
        _valid_split_ratio = (1.0 - split_ratio) / 2.0  
        
        # Define storage
        _train_data_list = []
        _validation_data_list = []
        _test_data_list = []
        
        _cnt = 0

        # Print header
        if verbose:
            print("Data statistics")
            print(f"{'Name':<10} | {'Total':<10} | {'Train':<10} | {'Valid':<10} | {'Test':<10}")
        # Create train validation and test sets
        for _key in data_dict.keys():
            # Grab data and shuffle it

            _size = len(data_dict[_key])
            random.shuffle(data_dict[_key])
            _sample_list = data_dict[_key]

            # Split it
            _train_split = _sample_list[: int(_train_split_ratio*(_size))] 
            _valid_split = _sample_list[int(_train_split_ratio*(_size)): int((_train_split_ratio+_valid_split_ratio)*(_size))]
            _test_split = _sample_list[int((_train_split_ratio+_valid_split_ratio)*(_size)) : ]
            
            # Verbose
            if verbose:
                print(f"{_key:<10} | {_size:<10} | {len(_train_split):<10} | {len(_valid_split):<10} | {len(_test_split):<10}")
        
            # Save them 
            _cnt += len(_train_split)
            _train_data_list = _train_data_list + _train_split
            _validation_data_list = _validation_data_list + _valid_split
            _test_data_list = _test_data_list + _test_split
        
        # Final statictics
        # Verbose
        if verbose:
            print(f"{'Dataset':<10} | {sum([len(_train_data_list), len(_validation_data_list), len(_test_data_list)]):<10} | {len(_train_data_list):<10} | {len(_validation_data_list):<10} | {len(_test_data_list):<10}")

        # Return values
        random.shuffle(_train_data_list)
        random.shuffle(_validation_data_list)
        random.shuffle(_test_data_list)
        return([_train_data_list, _validation_data_list, _test_data_list])
    


# Generate dataloader
def init_dataloader(data_loader_params:loader_params, dataset_info:data_params)->DataLoader:
    """
        Init of the  data loader. NOT TESTED FOR MULTIPLE GPU
        Creating wrapper arround data class. 

        ARGS:
            * batch_size, int, size of the batch
            * num_wokers, int, number of workers for data loading 
            * use_gpu, boolean, if gpu used
            * dataset_info --> data_params object

        Output:
            * Torch DataLoader
    """
    _ds = img_retreival_dataset(dataset_info)

    _dl = DataLoader(
        _ds,
        batch_size = data_loader_params.batch_size,
        num_workers = data_loader_params.number_of_workers,
        pin_memory = data_loader_params.use_gpu,
    )  
    return _dl