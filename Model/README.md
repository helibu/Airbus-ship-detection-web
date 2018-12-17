### Two models are used based on data analysis.

#### Ship existence model
One is for ship existence detection. An ensemble model combining a pretrained ResNet50 and a pretrained VGG16 is used here. The accuracy on both training and validation dataset are about 95%.


#### Ship location model
The other model is to locate ships on images. We used U-net with pretrained Inception v3 as encoder. The prediction results on Kaggle test data is about 23%.
