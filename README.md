# Airbus-ship-detection
http://www.shipdetection.com/

## Background and Data Analysis

This is a Kaggle challenge, here is the link. https://www.kaggle.com/c/airbusship-detection. The data size is about 30GB. The training data are masked
with a separate csv file. We need to submit the same mask csv file for the test
data. 

## Machine Learning Model
### 1. Ship existence model
We have developed an ensemble model, combining a pretrained resnet 50 and VGG16, to detect whether there is a ship in the image or not. The accuracy is about 95%. Considering the fact that users could upload lots of images in the same time and no ships on about 70% of the images, a simply model can speed up the whole process.

For ship-existence model, we refer to keras.application documents,
https://keras.io/applications/. We use VGG16 and ResNet50 pretrained with
ImageNet, add dense layer and flatten layer after the pretrained model. And
combin in Model folder.Loss function is
binary_crossentropy for we only need to detect if there is a ship or not. We
tried simple CNN at first, but it doesn’t work well. Then we used pretrained
models trained with imagenet. These models have been well trained for
classification purpose, which saved us a lot of time.

### 2. U-net model
We used a U-net model to predicte the location of the ships and draw a border-box around them.

For U-net model, referring to
https://github.com/qubvel/segmentation_models/tree/master/segmentation_models/unet . Based on their u-net model, we combine Inception v3 from
keras.application as encoder. U-net model
details could be found in Model folder.https://github.com/helibu/Airbus-shipdetection/tree/master/Model/U-net%20model. We use DICE/F1 score of the
intersection of predictions and ground truth based on the uneven distribution
of has-ship pixels and no ship pixels.
We choose U-net because most ships on images are quite small, and we need
a model that can handle high resolution information. So, we choose U-net.


## Web Application
### 1.web interface design
we have finished the framework of the web using html and css. 
### 2. backend
We used Flask to deploy our Python language machine learning model to our web application.
### 3. AWS elastic beanstalk
We use AWS elastic beanstalk service to run our web app, choosing python as the ec2 instance environment language, and also have applied for a domain name for our website. Our website address is: http://www.shipdetection.com/

## Features
### Home Page for image upload
This is the homepage of our website. We can upload a bunch of images in the same time, and see the image by click the "Images" button.

### Images Page to display uploaded and predicated images
In our image page, we can view our uploaded images with "has ship" or "no ship" on them, and if there is a ship on the image, the ship will be bordered by a red border-box. Also there is a "Clear images" button to clear the result.


### Sample page for sample images and sample download

### About page

