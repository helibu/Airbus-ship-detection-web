# Airbus-ship-detection

## model progress
### 1. CNN model for binary prediction
For now, we have developed an simple cnn model to detect whether there is a ship in the image or not. The accuracy is about 0.9. Considering the fact that users could upload lots of images in the same time and no ships on about 70% of the images, a simply model can speed up the whole process.
### 2. U-net model
We have deployed the U-net model, and can achieve predicting the location of the ships and draw a border-box around them.


## web progress
### 1.web interface design
we have finished the framework of the web using html and css styling. 

### 2. backend
We used Flask to deploy our Python language machine learning model to our web application.
### 3. AWS elastic beanstalk
We use AWS elastic beanstalk service to run our web app, choosing python as the ec2 instance environment language, and also have applied for a domain name for our website.
