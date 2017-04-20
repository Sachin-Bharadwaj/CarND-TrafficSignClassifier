#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Fig1.png "Visualization"
[image2]: ./examples/Fig2.png "TrainingSetDistbn"
[image3]: ./examples/Fig3.png "ValidationSetDistbn"
[image4]: ./examples/Fig4.png "TestSetDistbn"
[image5]: ./examples/BicycleLaws.jpg "German Sign-1"
[image6]: ./examples/BuildingSite.jpg "German Sign-2"
[image7]: ./examples/Kindergarten.jpg "German Sign-3"
[image8]: ./examples/Speedlimit50.jpg "German Sign-4"
[image9]: ./examples/Stop.jpg "German Sign-5"
[image10]: ./examples/Softmax.png "Top5-Softmax"
[image11]: ./examples/Visualization-Conv1.png "Conv1-Visualization"
[image12]: ./examples/JitteredImage.png "JitteredImage"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. **Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.**

I used the pandas/numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 data points
* The size of the validation set is 4410 data points
* The size of test set is 12630 data points
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. **Include an exploratory visualization of the dataset.**

Here is an exploratory visualization of the data set. It shows a random sample of images drawn from the training set

![alt text][image1]

Further, we look at the class distribution for each of the training/validation/test set

![alt text][image2]

![alt text][image3]

![alt text][image4]


###**Design and Test a Model Architecture**

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The data exploration excercise above shows that the training data is highly imbalanced. It implies that the network might get skewed towards the classes which have more representation in the training data set.
I decided to work with the given imbalanced data set to see what kind of performance is achievable. Later on towards the end of the notebook, I have also used the augmented data-set and the details about 
the generation of augmented data set and training is given at the end.

For pre-processing I decided to do subtract the mean of R,G,B channels across the dataset (for each training, validation and test set) and doing max-min normalization.


####2. **Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.**

My final model is the same as LeNet model except for the activation layer changed to ELU since it trains fasters. My primary aim was to see what kind of performance is acheivable with this simple architecture which consisted of the following layers:

| Layer         		|     Description	        					            | 
|:---------------------:|:---------------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							            | 
| Convolution 5x5     	| 6 filters, 1x1 stride, valid padding, outputs 28x28x6 	|
| ELU					|												            |
| Dropout               |                                                           |
| Max pooling	      	| 2x2 stride, valid padding,  outputs 14x14x6 				|
| Convolution 5x5	    | 16 filters, 1x1 stride, valid padding, outputs 10x10x16   |
| ELU                   |                                                           |
| Dropout               |                                                           |
| Max pooling           | 2x2 stride, valid padding, outputs 5x5x16                 |
| Fully connected-1		| 400x120        									        |
| ELU				    |        									                |
| Dropout	    		|												            |
| Fully connected-2   	| 120x84										            |
| ELU                   |                                                           |
| Dropout               |                                                           |
| Softmax               | 84x43                                                     |


####3. **Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.**

To begin with, I trained the model with ReLu activation and without dropouts, I searched for the hyperparameters , learning rate and regularization over small number of epochs say 5. Based on this search I concluded on 1e-3 as learning rate and 1e-3 as L2 regularization parameter.
At the same time, I realized that with ReLU activation some of the neurons might be dead in the starting based on bias initialization since relu activation is zero for negative input values, therefore I decided to try ELU activation instead.
Again training the same network but with ELU activation and with hyperparameters concluded earlier, I got validation accuracy ~91% and no further improvement in validation accuracy was noticed as number of epochs were increased.
In order to boost the accuracy further by getting the ensamble gain of the network and let the network generalize better, I decided to incorporate drop outs in the network. Again dropout being a hyperparameter was searched and a coarse value was found based on validation accruracy.

I used Adam Optimizer for training, batch size was set to 256 and number of epochs were varied from 50 to 100 depending upon the resource availability and training time required. Amazon web services are not always available and faced difficulty in using EC2 quite often.
So I decided to train for 50 to 100 epochs so that I could also run it on CPU with moderate training time requirements.

Finally, with network as described in the Point2 above and learning rate intially set to 1e-3, L2 regularization parameter=1e-3, dropout rate =10% (keep probability=0.9), with learning getting lowered to 1e-4 when the average validation accuracy over last 5 epochs is greater than 92%,
I was able to achieve average validation accuracy better than 93% without any data augmentation.

The Test accuracy was 91.8%

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of >93%
* test set accuracy of 91.8%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    Architecture was LeNet except for L2 regularization, dropouts and ELU activation change.
* What were some problems with the initial architecture?
    Without dropouts network was not generalizing to an accuracy better than 91% and with reLu the training was a bit slower for 0 bias initialization, so I decided to try ELU activation
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    Explained above
* Which parameters were tuned? How were they adjusted and why?
    Hyperparameters such as learning rate, L2 regualrization, dropout rate, were tuned over several iterations, by accessing the performance of validation set for different hyperparameters and choosing the one which yieds the best performance over small number of epochs and then training with the tuned hyperparameters for large number of epochs
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    Convolution layer are good at creating a heirarchical representation of the data. Since the data consist of images and images are built from low level features such as edges, circles and evolve into high level features heirarchically, deploying a convolutional layer is advantageous in this cases.
    Adding a drop out forces the network to learn alternate representation of the data, therefore network generalizes better and hence we get better accuracy.

If a well known architecture was chosen:
* What architecture was chosen? LeNet architecture was chosen as a base architecture due to its simplicity.
* Why did you believe it would be relevant to the traffic sign application? LeNet worked well for classifying hand written digits which are also images, so quite relevant for the current problem
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? Training accuracy is close to 99.8% , validation accuracy >93% but did not improve further with more epochs and neither with
data augmentation as shown in the end, this suggest that may be I need to add more conv layers/filters in order to increase the network capacity. Further test accuracy is close to validation accuracy. This suggest that the current network has been trained well.
However, I do feel that there is a lot of scope for improvement by adding more conv layer/more filters/trying out different color spaces or may be let the network learn it own color space. All of these suggestions require deeper networks and more training time and resources and has not 
been tried yet.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] 

The first image should be easy for the network to be classified

![alt text][image6] 

The second image contains mutiple signs, hence might be difficult for the network to be classified

![alt text][image7] 

The third image is not in the training set, hence may be difficult to be classified

![alt text][image8]
 
The fourth should be easy and should be classified

![alt text][image9]

The fifth also should be easy for classififcation


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| Comments
|:---------------------:|:---------------------------------------------:| -------------------
| Bicycle Laws      	| Right of way at next intersection      		| Correct
| Building Site			| General Caution 								| Wrong but close as network did classified as caution
| Kindergarten			| Wild animal crossing							| Wrong but again close as signs for kindergarten and wild animal crossing do look somewhat similar at a higher level 
| Speed limit 50		| Speed limit 100					 			| Wrong, network got it atleast right in the sense that it is a speed sign however not able to resolve difference b/w 50 and 100, may be a deeper network might be able to finer features better
| Stop			        | No Passing      							    | Wrong, this was surprise to me! network should have got it right, may be due to training on imbalanced class


Being pessimistic and not giving the benefit of doubt to the network, the model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This is no way close to the accuracy of the test set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][image10]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image11]

This is the feature map for the 1 convolutional layer for the 1st German Traffic Sign downloaded from web. As can be seen, these featue maps are able to detect the edges/gradients in the input image.

### Additonal Experiments - Training Data Augmentation

As explained above, the given training data is highly imbalanced, therefore additional fake dataset was generated. The function used to generate fake dataset is gen_augmented_data as given in the notebook. It includes small random rotations, translations, shearing and brightness change.

![alt text][image12]

I generated atleast 800 images for each class in the training set only. Further I added on more channel, GRAY channel in addition to the RGB channel to all the images. I trained the same network again with same hyperparmaters {no tuning of hyperparamaters done again in order to save time and compute}.
I was able to achieve >93% average validation accuracy within 15-20 epochs, however, no significant improvement in validation accuracy was observed until 50 epochs of training. Looks like network capacity needs to be increased now in order to improve validation accruacy. The test accuracy was also similar ~ 91.9%

