
# **Traffic Sign Classifier Project : Writeup report** 

---
## Introduction
**Abstract**

This traffic sign classifier project uses adapted tensorflow-based LeNet convolutional Neural Network to classify the german traffic signs. The raw dataset provided by Udacity was already split into training, validation and test set. However, the limited size of this dataset implied perfomance limitation, therefore translation, rotation, and zoom technique has been used for data augmentation. Moreover, grayscaling and normalization have been applied to the datasets to be used efficiently in the LeNet architecture. 

The final architecture proposed in this notebook relies on the original LetNet structure with additional dropout on fully connected layer and one additional convolution to avoid overfitting and increase the performed accuracy. Determination of the hyperparamaters was mostly experimental and made on rule of thumbs. 

The trained Neural Network performed a 97.39% accuracy on the validation set and a 96.68% accuracy on the test set. It performed 87.5% accuracy on the image found from the [web](https://github.com/olpotkin/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb
). 

**Rubric points** 

The rubric point of the project can be found [here](https://review.udacity.com/#!/rubrics/481/view). 

**Jupyter notebook** 

The project code can be found as a Jupyter notebook [here](./Traffic_sign_classifier.ipynb). 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project were the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

* Number of training examples provided = 34799
* Number of testing examples provided = 4410
* Number of testing examples provided = 12630
* Image data shape = 32 x 32 x 3
* Number of classes = 43

#### 2. Exploratory visualization of the dataset.
The table below lists the traffic signs included in the dataset.

|	ClassI |SignName	|
|:---------------------:|:---------------------------------------------:| 
|	0|Speed limit (20km/h)	|
|	1|Speed limit (30km/h)	|
|	2|Speed limit (50km/h)	|
|	3|Speed limit (60km/h)	|
|	4|Speed limit (70km/h)	|
|	5|Speed limit (80km/h)	|
|	6|End of speed limit (80km/h)	|
|	7|Speed limit (100km/h)	|
|	8|Speed limit (120km/h)	|
|	9|No passing	|
|	10|No passing for vehicles over 3.5 metric tons	|
|	11|Right-of-way at the next intersection	|
|	12|Priority road	|
|	13|Yield	|
|	14|Stop	|
|	15|No vehicles	|
|	16|Vehicles over 3.5 metric tons prohibited	|
|	17|No entry	|
|	18|General caution	|
|	19|Dangerous curve to the left	|
|	20|Dangerous curve to the right	|
|	21|Double curve	|
|	22|Bumpy road	|
|	23|Slippery road	|
|	24|Road narrows on the right	|
|	25|Road work	|
|	26|Traffic signals	|
|	27|Pedestrians	|
|	28|Children crossing	|
|	29|Bicycles crossing	|
|	30|Beware of ice/snow	|
|	31|Wild animals crossing	|
|	32|End of all speed and passing limits	|
|	33|Turn right ahead	|
|	34|Turn left ahead	|
|	35|Ahead only	|
|	36|Go straight or right	|
|	37|Go straight or left	|
|	38|Keep right	|
|	39|Keep left	|
|	40|Roundabout mandatory	|
|	41|End of no passing	|
|	42|End of no passing by vehicles over 3.5 metric tons	|

The figure below shows the amount of data to each ClassID. It shows that there are more images for some instance than the others (e.g. instance 19 has less then 100 training sample). The Neural Network will perfom better for the classes with higher number of training data. Thus, data augmentation is necessary to improve the accuracy of those classes with lower number of samples. Intuitively, I would say that if the variable repartition of the number of traffic per class will bias the neural network ; the probability is a function of the number of instance.

![data set visual](images/dataset_visu.png)

## Design and Test a Model Architecture

#### 1. Proprocessing the dataset

Two preprocessing methods have been used in this classifier : grayscaling and normalization. The traffic signs color is not as relevant as their shape, therefore using black and white images provides lower calculation for nearly same results. Normalization of the images makes the calculations easier for network. 

To increase the data set, I have choosen to rotate + zoom randomly (10° max, 1.25 zoom) and translate randomly in x and y direction (6 pixels direction max) the training data. It has indeed improved the accuracy of the Neural Network without changing the hyperparamaters of the network. 

As further experimentation I would propose to change the contrast of the dataset, and use more data augmentation techniques found on the web such as adding noise and equalizing the data repartition. The images below illustrate the preprocessing and the data augmentation achieved compared to the initial data. 

![Preprocessed data](images/process_image.png)

![data set visual augm](images/dataset_visu_augm.png)

#### 2. Final model architecture

| Layer         		|     Description	        					| Output |
|:---------------------:|:---------------------------------------------:|:----:|
|	Input	|	32x32x1 Grayscale images	|	|
|	Convultion 1 	|	5x5 filter , 1x1 stride, VALID padding	|	28x28x6
|	RELU	|	Activation	|	28x28x6
|	Max pooling	|	2x2 size, 2x2 strides 	|	14x14x6
|	Convultion 2 	|	5x5 filter , 1x1 stride, VALID padding	|	10x10x16
|	RELU	|	Activation	|	
|	Max pooling	|	2x2 size, 2x2 strides 	|	5x5x16 
|	Convultion 3 	|	2x2 filter , 1x1 stride, VALID padding	|	4x4x450
|	RELU	|	Activation	|	|
|	Max pooling	|	2x2 size, 2x2 strides 	|	2x2x450 
|	Flatten layer	|	Convert array to list 	|	1800
|	Fully connected 1	|	1800x120 matrix multiplication	|	120
|	RELU	|	Activation	|	120
|	Dropout	|	Keep probability of 0,5 	|	120
|	Fully connected 2	|	120x84 matrix multiplication	|	84
|	RELU	|	Activation	|	84
|	Dropout	|	Keep probability of 0,5 	|	84
|	Fully connected 3	|	Output ClassID	|	43

#### 3. Problem approach and model training

LeNet architecture is known to be a good framework for image classification, therefore it was used as a first experiment. AdamOptimizer is also a commom use, however another choice might have be the MomentumOptimizer. The raw LeNet architecture with unaugmented dataset perfomed a 80% ~ <90% (validation set) accuracy. Getting more training data and adding dropout layers allowed going from 92% to 94% (validation set). Last optimization made was to add a convolution layer with a smaller filter size and large filter depth to improve accuracy. Thanks to this layer, accuracy went above 94%. 

Classical hyperparameters have been used in this classifier, learning rate = 0.001, dropout keep probability = 0.5, EPOCHS < 25, batch size of 128. I used my CPU for training the dataset, which was time consuming for parameter investigation. I applied small variation of the paramaters to finetuning the parameters but it didn't show huge performance changes. Therefore more investigation on more powerfull GPU instance must be done in order to estimate the impact of each parameter. The effect of the dropout is however powerfull and adding a convolutional layer for goind deeper was an effective choice. 

I used graphical representation to see the convergence of the network (how fast, how accurate, convergence), this process make easier the comparison of several hyperparameter sets. However, prediction probability analysis must also be considered when talking about the quality of the proposed solution. 

### Test a Model on New Images

#### 1.New image data set

I took the images from the [project](https://github.com/olpotkin/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb
) of Oleg Potkin (see the link above) because it was faster and I did agree with his given prediction difficulties. 

![Web images data set](images\web-dataset.png)

(copy paste from his project) 

* The image #1 might be difficult to classify because it's rotated

* The image #2 might be difficult to classify because it has complex background

* The image #3 might be difficult to classify because it has noizy background

* The image #4 might be difficult to classify because it has unusual color of sign's border (looks like pink)

* The image #5 might be difficult to classify because it has noizy background

* The image #6 might be difficult to classify because it has noizy background

* The image #7 might be difficult to classify because background contains 2 parts

* The image #8 might be difficult to classify because it's rotated

I applied the same preprocessing pipeline to the new image data set. The result is given below : 

![Webimagepreproc](images\web-dataset-preproc.png)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

###### 2.1 Without extra convolutional layer

The model without the extra convolutional layer provides an accuracy on the new images of 87,5%, the table below shows the results. It is normal that it performs worse than the validation and test set evaluation, but is however an acceptable prediction ; it means that the network extracts features correctly.  If we compare to the histogram of the data given above, it is obvious that the quality of the prediction is related to the number of images available for particular instance. For example, the no-passing signal has ~4000 training images, and provides a prediction probability of 1. To confirm this assumption, the size of the training set of the traffic signal sign is small (<1000), and the prediction is wrong. Worst, the correct probability of 0.0648 isn't near to be correct. 

Thus, the more data the better the prediction is. Despite the limitation of the data size we must verify if the network extract feature correctly. 

| Image			        |     Prediction	        					| Probability | ClassID pred. - Correct | Correct pred. prob | 
|:---------------------:|:---------------------------------------------:|:------------:|:--------:|:------------------------: 
| General caution     		| General caution							| 0.843 | 18 | -
| No passing    			| No passing								| 1.000 | 9  | -
| Speed limit (20km/h)					| Speed limit (20km/h)			| 0.564 | 0  | - 
|!  <b>Traffic signal    </b> ! 		| <b>General caution	</b>	| 0.588 | 18 - 26 | 0.0648 
| Stop			| Stop                                                  | 1.000 | 14 | -
| Priority road				| Priority road								| 0.999 | 12 | -
| Yield   		| Yield					 			                 	| 0.977 | 13 | -
| Turn right			| Turn right                                    | 0.994 | 33 | -

###### 2.2 With extra convolutional layer

#####  Filter depth  = 450, result 87.5% on new images
![test](./images/Accuracy_evolution/ac_evol_CONV_450_15_0.5_0.001.png)

| Image			        |     Prediction	        					| Probability | ClassID pred. - Correct | Correct pred. prob | 
|:---------------------:|:---------------------------------------------:|:------------:|:--------:|:------------------------: 
| General caution     		| General caution							| 1.000 | 18 | -
| No passing    			| No passing								| 1.000 | 9  | -
| Speed limit (20km/h)					| Speed limit (20km/h)			| 1.000 | 0  | - 
|!  <b>Traffic signal    </b> ! 		| <b>General caution	</b>	| 1.000 | 11 - 26 | 0 
| Stop			| Stop                                                  | 1.000 | 14 | -
| Priority road				| Priority road								| 1.000 | 12 | -
| Yield   		| Yield					 			                 	| 1.000 | 13 | -
| Turn right			| Turn right                                    | 1.000 | 33 | -

* Dropout technique provide a starting prediction of 0.8 
* The network converge 
* After EPOCHS 11 the network starts oscillating
* Prediction of 97.39% and 96.68% (validation and test set respectively)
* The Network does not predict every signs of the new image set
* However its prediction is certain for the rest of the images. But really wrong with the one it can not predict

##### Filter depth  = 250, result 87.5% on new images

![test](./images/Accuracy_evolution/ac_evol_CONV_250_15_0.5_0.001.png)

| Image			        |     Prediction	        					| Probability | ClassID pred. - Correct | Correct pred. prob | 
|:---------------------:|:---------------------------------------------:|:------------:|:--------:|:------------------------: 
| General caution     		| General caution							| 1.000 | 18 | -
| No passing    			| No passing								| 1.000 | 9  | -
| Speed limit (20km/h)					| Speed limit (20km/h)			| 0.8 | 0  | - 
|!  <b>Traffic signal    </b> ! 		| <b>General caution	</b>	| 0.4 | 11 - 26 | 0 
| Stop			| Stop                                                  | 1.000 | 14 | -
| Priority road				| Priority road								| 1.000 | 12 | -
| Yield   		| Yield					 			                 	| 1.000 | 13 | -
| Turn right			| Turn right                                    | 1.000 | 33 | -



* With smaller filter depth the final predictions on validation and test sets are smaller
* The predictions probability are not as high as the previous network
* Let's see the effect of the dropout

##### Filter depth  = 450, result 87.5% on new images

![test](./images/Accuracy_evolution/ac_evol_CONV_450_15_0.75_0.001.png)

| Image			        |     Prediction	        					| Probability | ClassID pred. - Correct | Correct pred. prob | 
|:---------------------:|:---------------------------------------------:|:------------:|:--------:|:------------------------: 
| General caution     		| General caution							| 1.000 | 18 | -
| No passing    			| No passing								| 1.000 | 9  | -
| Speed limit (20km/h)					| Speed limit (20km/h)			| 0.5 | 0  | - 
|!  <b>Traffic signal    </b> ! 		| <b>General caution	</b>	| 1.000 | 24 - 26 | ~0 
| Stop			| Stop                                                  | 1.000 | 14 | -
| Priority road				| Priority road								| 1.000 | 12 | -
| Yield   		| Yield					 			                 	| 1.000 | 13 | -
| Turn right			| Turn right                                    | 1.000 | 33 | -



* Dropout allows us to start at higher prediction on validation set
* However this model does not provides similar results as the first 
* The curve starts high but does not seem to increas after many epochs
* Let's keep the depth of 450 but decrease the dropout and the learning rate 

##### Filter depth  = 450, result 87.5% on new images

![test](./images/Accuracy_evolution/ac_evol_CONV_450_15_0.625_0.0008.png)

##### Filter depth  = 450, result 87.5% on new images

![test](./images/Accuracy_evolution/ac_evol_CONV_450_15_0.56_0.0008.png)

| Image			        |     Prediction	        					| Probability | ClassID pred. - Correct | Correct pred. prob | 
|:---------------------:|:---------------------------------------------:|:------------:|:--------:|:------------------------: 
| General caution     		| General caution							| 0.925 | 18 | -
| No passing    			| No passing								| 1.000 | 9  | -
| Speed limit (20km/h)					| Speed limit (20km/h)			| 0.5 | 0  | - 
|!  <b>Traffic signal    </b> ! 		| <b>General caution	</b>	| 0.549  | 18 - 26 | ~0 
| Stop			| Stop                                                  | 1.000 | 14 | -
| Priority road				| Priority road								| 1.000 | 12 | -
| Yield   		| Yield					 			                 	| 1.000 | 13 | -
| Turn right			| Turn right                                    | 1.000 | 33 | -

* Still the first model perform better
* Let's try when getting closer to the first network parameters and decrease the learning rate to stabilize the curve after EPOCHS 10
* It is always the same sign that is misclassified. As meantioned earlier I suspect a lack of training data for this particular instance. From now on I do not expect from the network to classify correctly. However, tbe network must at least 

##### Filter depth = 450, result 87.5% on new images

![test](./images/Accuracy_evolution/ac_evol_CONV_450_25_0.54_0.00095.png)

##### Filter depth = 450, result 87.5% on new images

![test](./images/Accuracy_evolution/ac_evol_CONV_450_25_0.42_0.0009.png)

* Decreasing the dropout probability, doesn't provide any furtger evoltuion. 

### Final parameters

As shown by the graphics above, the best performances are achieved with the parameters set of : 
* Filter depth 450
* Dropout 0.5
* EPOCHS 15 - (11 should do) 
* learning rate 0.001

### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Tensorflow library does of an internal problem with the functions that provides the visual output ; it does not work every time.
however, here is an example of a process visual.
![](./images/Evolution_fig/05-conv1.png)

![](./images/Evolution_fig/06-conv1_relu.png)

![](./images/Evolution_fig/07-conv1_pool.png)

### Summary 
* It is a long process to collect, classsify, preprocess, etc. images for deep learning. Understanding the complexity of the approach allows one to choose appropriatly between classical coding and Neural Network. 
* A lot of situations might be treated by a machine learning approach, however without a sufficient dataset it is a no go. Collect and classify the data is a tremendous work by itself. 
* Even though it exists a lot of documentation on Neural Network, every project will be different with a particular behavior and different fine tuning parameters
* Training the network requires a huge amount of calculation, even though there are powerful computation capabilities I wouldn't be suprised complex cases are not achievable 
* Finding the appropriate architecture is hard and this project illustrates why machine learning is still maturing
* Fine tuning the parameters is done by experimentation. Since the calculation is long it is tricky and time consuming to find the most appropriate set of parameters. 
