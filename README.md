# Airbus_ship_segmentation

This is a repository for a ROBT407 - Machine Learning course. The project's goal is to construct a model that performs well on (Airbus Ship Detection)[https://www.kaggle.com/c/airbus-ship-detection] dataset.

For this dataset, one must perform an image segmentation, i.e. give a mask that describes the location of the ships.

The project was done by Nuradil Kozhahmet, Saniya Abushakimova and Danel Batyrbek.

## Airbus Ship Detection Dataset

Airbus Ship Detection dataset is provided by Airbus. It consists of roughly 200000 satelitte images which may or may not contain a ship. There are mostly photos of a water surface, but sometimes it is a land photo. There may be clouds or something else.

The segmentation information is given in the form of run pixel encoding, i.e. if the image is flattened, then by starting in the `i`th pixel, `j` pixels belong to a mask. Thus, the data is given in the form of `i j` pairs.

## Toolset
For this project we used several toolsets. Standard:
- numpy
- pandas
- matplotlib.pyplot

Additional important:
- pytorch (i.e. `torch`, `torchvision`)
- OpenCV methods (`opencv-python`)
- skimage
- fastai

For *.ipynb we used standard Jupyter Notebook.

## DNN Architectures
For this project we used different models, such as
- ResNet
- UNet
- AlexNet
- SqueezeNet
