# Balancing Performance and Interpretability in Medical Image Analysis: Case study of Osteopenia

<br/>
<br/>

This is official repository of our paper named in title.

<br/>

![result image](https://github.com/mikulicmateo/osteopenia/blob/master/media_readme/gradcam_3.png?raw=true)

### Abstract

Multiple studies within the medical field have highlighted the remarkable effectiveness of using convolutional neural networks for predicting medical conditions, sometimes even surpassing that of medical professionals.
Despite their great performance, convolutional neural networks operate as black boxes, potentially arriving at correct conclusions for incorrect reasons or areas of focus.
Our work explores the possibility of mitigating this phenomenon by identifying and occluding confounding variables within images.
Specifically, we focused on the prediction of osteopenia, a serious medical condition, using the publicly available GRAZPEDWRI-DX dataset.
After detection of the confounding variables in the dataset, we generated masks that occlude regions of images associated with those variables.
By doing so, models were forced to focus on different parts of the images for classification.
Model evaluation using F1-score, precision, and recall showed that models trained on non-occluded images typically outperformed models trained on occluded images.
However, a test where radiologists had to choose a model based on the focused regions extracted by the GRAD-CAM method showcased different outcomes.
The radiologists' preference shifted towards models trained on the occluded images.
These results suggest that while occluding confounding variables may degrade model performance, it enhances interpretability, providing more reliable insights into the reasoning behind predictions.

<br/>
<br/>

## Minimum working code example

<br/>

### Clone repository

To test our minimal working sample please clone this repository:
```bash
git clone https://github.com/mikulicmateo/osteopenia.git
```

### Download the models

Download our models from url. 
Place the models in `OsteopeniaMinimumWorkingExample/models`.

### Preparing environment

<br/>

You can create virtual environment using command:
```bash
python3 -m venv venv
```

Don't foget to source it:
```bash
source venv/bin/activate
```

We prepared `minimum_requirements.txt` to get you started as soon as possible.
Just run:
```bash
pip install -r minimum_requirements.txt
```

### Testing

<br/>

To test our minimum working code example just load `OsteopeniaMinimumWorkingExample/ExampleNotebook.ipynb`.

![output image](https://github.com/mikulicmateo/osteopenia/blob/master/media_readme/output.png?raw=true)

<br/>
<br/>

## Reproduce our results

### Clone repository

To test our minimal working sample please clone this repository:
```bash
git clone https://github.com/mikulicmateo/osteopenia.git
```

<br/>

### Download the dataset

Download original [dataset][1]. (Dataset paper:
Nagy, E., Janisch, M., Hržić, F. et al. A pediatric wrist trauma X-ray dataset (GRAZPEDWRI-DX) for machine learning. Sci Data 9, 222 (2022). https://doi.org/10.1038/s41597-022-01328-z)

<br/>

### Preparing environment

<br/>

You can create virtual environment using command:
```bash
python3 -m venv venv
```

Don't foget to source it:
```bash
source venv/bin/activate
```

We prepared `minimum_requirements.txt` to get you started as soon as possible.
Just run:
```bash
pip install -r all_requirements.txt
```

<br/>

### Generating filtered dataset

<br/>

Extract all the images from (downloaded) original dataset to `dataset/images` folder.
Extract all the `*.json` annotations from (downloaded) original dataset found in `folder_structure.zip` under `supervisely/wrist/ann` to `dataset/annotations_all` folder.

<br/>

Your **dataset** folder should now look like this:
```
|---dataset
    |---annotations_all
    |   |---0001_1297860395_01_WRI-L1_M014.json
    |   |---...
    |    ...
    |---images
        |---0001_1297860395_01_WRI-L1_M014.png
        |---...
        ...
```
<br/>

**Edit** `config.json` with **your paths** for `osteopenia_dataset_csv_path` and `additional_annotations_path`.

<br/>

Change directory to metadata:
```bash
cd metadata
```

Run `create_filtered_dataset_files_and_csv.py` script to generate new filtered dataset:
```bash
python3 create_filtered_dataset_files_and_csv.py
```
Next, run `mask_dataset_images.py` to create masks ("dummy" and real) for each image in filtered dataset:
```bash
python3 mask_dataset_images.py
```

<br/>


![mask image](https://github.com/mikulicmateo/osteopenia/blob/master/media_readme/mask_mosaic.png?raw=true)

<br/>
<br/>

### Run training

<br/>


Change directory to main project directory:

```bash
cd ..
```

Run `train_all.py` script to train all models.
```bash
python3 train_all.py
```

<br/>

![models image](https://github.com/mikulicmateo/osteopenia/blob/master/media_readme/test-mosaic-1.png?raw=true)



[1]: https://www.nature.com/articles/s41597-022-01328-z?fbclid=IwAR35HKVMkNo2ARi3KgZuP3Inv9P7UpjmalDrUj0oa57_Y5bvXHHCFVp-1Ig
