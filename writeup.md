#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/pct_signs_by_label.png "Data distribution"
[image2]: ./examples/dark_sign.png "Dark sign"
[image3]: ./examples/equalization.png "Equalization examples"
[image4]: ./examples/munich1.png "Traffic Sign 1"
[image5]: ./examples/munich2.png "Traffic Sign 2"
[image6]: ./examples/munich3.png "Traffic Sign 3"
[image7]: ./examples/munich4.png "Traffic Sign 4"
[image8]: ./examples/munich6.png "Traffic Sign 5"

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of the validation set is 4410 images
* The size of test set is 12630 images
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I quickly scanned through the data to assess the quality and character of it.  I found two problems.

First, the classes are extremely unbalanced, with the largest class having 2,010 images, and the smallest classes having less than 1/10th as many items in the training set: 180.  As well, more than 80% of the images were assigned to the top 50% of classes.  The chart below

![Distribution of training data by classes][image1]

I also randomly displayed about 100 signs (though this is not reflected in the final version of my notebook.)  It was painfully obvious that a lot of the images were dark, low contrast, or just difficult to read.  A challenge!

![dark, low contrast sign][image2]

Based on that, I decided to try out three ways of improving the data:

1) Histogram equalization to increase contrast and so that each sign made full use of the available dynamic range.
2) Rebalancing classes to have 2010 images each.
3) In the act of rebalancing, creating new images for underrepresented classes.

I will detail that in the next section.

I noticed later that many people have made their images grayscale.  Good idea, something I should have implemented but did not.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For equalization, I made use of the OpenCV library, trying out the function `cv2.equalizeHist` and the function `cv2.createCLAHE`.  The former is a straight forward equalization function, whereas the latter allows equalizing different regions of the image.  Below is an image giving a sign, the output of `cv2.equalizeHist`, and the output of `cv2.createCLAHE(clipLimit=4, tileGridSize=(3, 3))`:

![Equalization examples][image3]

I played with a few parameters and I decided that the setting above for `cv2.createCLAHE` most improved the image while keeping a natural look and feel.  My reasoning was that, if the image looked natural, while being high contrast, this would keep most of the information while making it easier for convolutions to pick up features.  In retrospect, having browsed a few other solutions, I think now that my reasoning was wrong, and a very high contrast grayscale image that had unnatural looking edges might have performed better.

I then rebalanced classes by adding transformed examples of the existing examples.  In other words, I took the signs in the underrepresented classes and added rotated and warped versions of these signs to the dataset until each class had 2010 members.  To do this I initially looked for an existing library or implementation, and I chanced on [Alex Staravoitau's solution for this exact project](https://navoshta.com/traffic-signs-classification/).  I made use of Alex's code, adding two of his functions to `tools.py` for warping and rotating signs by random amounts, and using them to add examples to my training set.  Alex's solution was very impressive, with 99.3% accuracy, and given more time, I would certainly like to investigate some of his approaches more deeply.

Finally, I normalized pixel values to the interval [-1, 1] to improve training.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6   				|
| Convolution 5x5x6	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| Fully connected		| Flattened prev layer to 400, outputs 120      |
| RELU                  |                                               |
| Fully connected       | 120 inputs to 84 outputs                      |
| RELU                  |                                               |
| Dropout               | keep_prob set from 0.3 to 0.9 in training     |
| Fully connected       | 84 inputs to 43 outputs                       |
| Dropout               | keep_prob set from 0.3 to 0.9 in training     |
| Softmax				| on logits to determine class                  |

I hewed pretty close to the LeNet architecture suggested by the project intro.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I explored the hyper-parameters:

- epochs: 10
    - I explored various numbers of epochs, from as little as 3, for fast experimentation, to as many as 50.  Ultimately, I found that the set of parameters chosen trained best on a few epochs, around 10, and often topped out before that.
    - As the reviewer suggests, early stopping is a good idea here, and something I look forward to implementing in Project 3.
- batch size: 64
    - I had luck with batch sizes of 64 and 128.  Smaller and my model often failed to train successfully, sometimes even at 64.  I did not really experiment with going larger.
- learning rates: 0.002
    - I had luck with rates even as high as 0.05, which I think is partly due to the adaptive nature of the optimizer.  However, with low learning rates like 0.0001, the model seemed to stop improving well before 0.90 validation accuracy, so I abandoned that approach.
- keep probability for dropout layers
    - I used an adaptive approach here.  My model uses 0.3 as a keep_prob until it reaches 0.9 accuracy, when it switches to 0.4.  It then goes on to increase slightly, until it reaches 0.94, where I hold it at 0.6.
    - I tried a variety of keep_probs.  It's a reach to make a generalization here, but I did notice that low numbers like 0.3 tended to improve quickly at first, then slow down and struggle to move past 0.9 ish.  On the other hand, it seemed that low or zero dropout had the opposite problem, so I tried the adaptive approach above.
- adding balanced classes or leaving X_train unaltered
    - Ultimately, I went with balanced classes using the techniques described above.  However, I did not see a significant difference from this in my model's performance on the *validation* set.  I had models that performed similarly to my final model that used unbalanced training data.
    - It could be that it did improve the model's generality and that by going with balanced training data, my model generalized better to the *test* set.  However, I didn't check that, as I didn't want to implicitly overfit the test data.
- sigma value for weight initialization: 0.1
    - I had a lot of trouble with setting the input weights.  When I left it higher, at 0.2 or 0.5 or even 1, the network started off with large dead zones with zero gradient.  When I tried it much smaller than 0.1, for example 0.01, the network seemed to learn very slowly and often did not seem to gain any traction.
    - With ∞ time, I would like to try smaller `sigma` and a higher learning rate and see what happened.
- I stuck with the AdamOptimizer that was used in the LeNet solution for handwriting recognition.
- Last, I experimented with using dropout or not in the final two layers.  I noticed significant improvements in training speed by adding dropout on the last layer, and again by adding it to the second-to-last layer.  So I settled on dropout on the last two layers as a topological hyperparameter.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.940
* test set accuracy of 0.9303

My final 

If a well known architecture was chosen:
* What architecture was chosen?
    - I chose to modify the LeNet architecture for handwriting recognition.
* Why did you believe it would be relevant to the traffic sign application?
    - Since it was suggested, I thought it was a good starting point.  I did not modify it much, though I did add dropout to the last two layers in training, and I tuned that parameter a bit.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    - I'm not sure how well the model is working. I struggled a bit to get it to train at all in the sense that, when I had set `sigma`, the standard deviation for the weights, higher, the model could not do significantly better than random prediction.
    - I started looking through the network and it seemed as if many parts of the network were not being used, and I came to realize that I needed to lower `sigma` so that the Relu layers did not combine with large weights to create dead portions of the network.
    - That helped, but I still find the network does not train as reliably as I would like, results seem to vary a lot by training attempt.
    - I'm also struggling with understanding exactly how sessions, graphs, and savers interact.  It seems as if sometimes I am using them incorrectly, and thus I am not restoring the trained model correctly.
    - It actually seemed to me that a high dropout rate led to better training.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I think these are all relatively easy images to classify, and indeed the model correctly classified all of them with certainty `100%, 69%, 100%, 86%, 100%` respectively.  These are screen shots of München (Munich) street signs from Google Street View.  As such, they are fairly high quality.

Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Turn Left Ahead       | 100% Turn Left Ahead                          | 
| Yield                 | 69% Yield                                     |
| No Entry              | 100% No Entry                                 |
| Ahead Only            | 86% Ahead Only                                |
| Priority Road         | 100% Priority Road                            |

The top 5 softmax probabilities and the corresponding labels are:

```
[[  1.00000000e+00,   4.72971034e-14,   5.65832324e-15, 3.49641945e-15,   1.87820609e-16],
[6.91244841e-01,   8.29834789e-02,   2.74962969e-02, 2.62410603e-02,   2.17501931e-02],
[1.00000000e+00,   1.56135863e-23,   9.47764811e-29, 2.41173133e-29,   1.02852887e-30],
[8.57601285e-01,   1.36957660e-01,   3.63318971e-03, 1.50579948e-03,   1.23508711e-04],
[9.99998093e-01,   1.85761780e-06,   2.49864893e-08, 1.37784284e-10,   9.75353687e-11]]

[[34, 35, 37, 38, 36],
[13, 29, 41, 28, 22],
[17, 14, 20, 16, 26],
[35, 36, 37, 34, 38],
[12, 15, 26, 17, 13]]
```

The 2nd sign has, overlaid on it, another small "no stopping" sign, which could account for the networks uncertainty.  No Stopping signs are not in the dataset, so it's hard to know how the network would be affected.

The 4th sign is half in shade, half in light, and this could cause some difficulty.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

See above.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

See above.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


