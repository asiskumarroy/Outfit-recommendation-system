# Outfit-recommendation-system
A CNN based approach to recommend similar outfit based on a outfit chosen by the user

### Dependancies:

Pytorch 1.5.1
Cuda 10.2 with cudnn 7.6.5
numpy,pandas and matplotlib recent versions

### Dataset

The dataset used here is a part of the big DeepFashioin dataset.Due to memory limit issues of git the entire dataset on which the model is trained
couldn't be updated.However the dataset can be easily downloaded from the internet.

### About the model

This is a outfit recommendation system where instead of using traditional reinforcement learning algorithms like in the e-commerce websites
,convolutional neural networks are being used for recommending a set of images to the user.The model uses the ResNet-152 architecture trained on the dataset.
The user chooses a set of images as per the weather and the event he is attending and then that image is passed through a the trained CNN model and the output of the first
FC layer of the trained architecture is extracted to find out the "cosine-similarity" of the chosen outfit with other present images similarly feed-forwarded through the
network.The top-6 outfits with highest cosine similarity are being recommended to the user.

The details of images in CSV file is not as per the images provided in the data and test directory as only a part of the original data on which the model is trained 
has been updated as an example.
