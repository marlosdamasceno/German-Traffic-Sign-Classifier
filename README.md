## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

# **Marlos Damasceno** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Output/all_possible_labels.png "All images labels"
[image2]: ./Output/train_histogram.png "Training labels histogram"
[image3]: ./Output/valid_histogram.png "Validating labels histogram"
[image4]: ./Output/test_histogram.png "Testing labels histogram"
[image5]: ./Output/pre_process.png "Before/After gray scale"
[image6]: ./Output/images_from_web.png "Images from Google Street View"
[image7]: ./Output/images_from_web_predictions.png "Top five softmax probabilities"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Files Submitted

#### 1. The project submission includes all required files.

Writeup Markup file that you are reading it!
[Jupyter Notebook Code]()

[HTML of Jupyter Notebook]()

### Dataset Exploration
#### 1. The submission includes a basic summary of the data set.

I used the matplotlib.pyplot library to plot each first image of each label of training set. Moreover, I ploted the histogram for each data set, training, validating and testing.
To get some basic information as below I worked with len() and numpy.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. The submission includes an exploratory visualization on the dataset.

As stated above, here is all possible labels (43) that is on the training set.
![alt text][image1]
Moreover, some stats with histogram.
![alt text][image2]
![alt text][image3]
![alt text][image4]


### Design and Test a Model Architecture

#### 1. The submission describes the preprocessing techniques used and why these techniques were chosen.

As a first step, I decided to convert the images to gray scale because removes the effect of brightness other factors that a color image have.

Image before and after gray scaling.

![alt text][image5]

Moreover, I normalized the image data to have mean around zero.


#### 2. The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

My final model consisted of the following layers:

| Layer         		|     Description	        		| 
|:-----------------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Image   				| 
| Convolution 5x5     		| 1x1 stride, valid padding, outputs 28x28x16	|
| RELU				|						|
| Max pooling	      		| 2x2 stride, same padding, outputs 14x14x40 	|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x40 	|
| RELU				|						|
| Max pooling	      		| 2x2 stride, same padding, outputs 5x5x40 	|
| Flatten	      		| outputs 1x1000			 	|
| Fully connected		| inputs 1000, outputs 1000			|
| RELU				|						|
| Dropout			| probability 0.5				|
| Fully connected		| inputs 1000, outputs 400			|
| RELU				|						|
| Dropout			| probability 0.5				|
| Fully connected		| inputs 400, outputs 120			|
| RELU				|						|
| Fully connected		| inputs 120, outputs 84			|
| RELU				|						|
| Fully connected output	| inputs 84, outputs 43				|
|				|						|
 


#### 3. The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyper parameters.

To train the model, I used an LeNet architecture adding two dropouts.
The optimizer it was the Adam Optimizer, because it is enough for this case. The batch size was 512 to speed up training. Epochs was 15, enough to a high accuracy without taking to long to train.
For the learning rate I use 0.001, for the dropout 0.5 of the probability to keep the neurons. Moreover, to initialize the weights I use 0 for the mean and 0.1 for the standard deviation.
In resume:
Architecture: LeNet
Batch size: 512
Epochs: 15
Learning rate: 0.001
Dropout: 0.5
Mu: 0
Sigma: 0.1

#### 4. The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.

My final model results were:
* validation set accuracy of 0.948 or 94.8%
* test set accuracy of 0.936 or 93.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? LeNet, because I had it done, it was just improve it.
* What were some problems with the initial architecture? Did not achieve 93% of accuracy, only got 86%
* How was the architecture adjusted and why was it adjusted? I added to dropouts, they improved about 5%
* Which parameters were tuned? How were they adjusted and why? I increased the batch size to be quicker and tuned the dropout, the rest remained the same.
* What are some of the important design choices and why were they chosen? The dropout it is a random way to take some neurons off the net work and that improves the learning of the neural network because it not remains on particulars neurons, it has to work well for any case. 

### Test a Model on New Images

#### 1. The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.

Here are five German traffic signs that I found on the web using Google Street View:

![alt text][image6]

I choose them because they give me some interest characteristics, such as the *Speed limit (30km/h)* is one of the most popular image on the training set with 1980 samples. Against of *Go straight or right* that does not have that many on the training set, only 330. Moreover, *Road work* (1350) it seems a difficult one to train, because can be confused with many other types of signs. *Keep right* has 1860 and *Priority road* has 1890.


#### 2. The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        	| 
|:-----------------------------:|:-------------------------------------:| 
| Priority road			| Priority road				| 
| Speed limit (30km/h)		| Speed limit (30km/h)			|
| Road work			| Road work				|
| Go straight or right    	| Ahead only				|
| Keep right			| Keep right      			|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%, compared with the test accuracy it is about 13% less.

#### 3.The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.
Here are the top five softmax probabilities:
![alt text][image7]

