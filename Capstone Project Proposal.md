Machine Learning Engineer Nanodegree Capstone Proejct
Google AI Open Images - Object Detection Track

Matthew Lee
7/21/2018

Content

    Domain Background
    Problem Statement
    Datasets and Inputs
    Solution Statement
    Benchmark Model
    Evaluation Metrics
    Project Design

Domain Background

In recent years, one of the hottest application of AI in computer vision has been object detection. Object detection has wide variety of applications such as robotics, autonomous driving, facial recognition, security, and etc. Such advancements are transforming our society and helping us to achieve higher goals with unprecedented speed and accuracy.

Despite some of the hypes around AI's unlimited potentials, I believe current AI algorithms and solutions are very specific to individual applications. However, I believe object detection algorithm is an example in which it can be used across multiple applications.

This capstone project wishes to expand obejct detection domain by developing an algorithm to successfully detect objects in given image dataset.


Back To Top

Problem Statement

Very popular method to develop an object detection algorithm is achieved by training neural networks with labeled images. Carefully chosen neural network architecture will be able to surpass human's performance, but training process requires enormous amount of data. We will utilize plethora of images with labels from Open Images Dataset, publicly released by Google AI, to develop an object detection algorithm in order to push the limits of object detection capability. Once trained, we will be able to monitor trained algorithm's performance by evaluating computed mean Average Precision (AP).


Back To Top

Datasets and Inputs

In this project, Open Images Dataset, publicly released by Google AI, will be used for training the algorithm. Dataset contains 1.7 million images that contain very diverse and multiple objects. Across images in the dataset, there are 12 million bounding-box annotations for 500 object classes, making it the largest existing dataset with object location annotations. Bounding-box annotations are drawn by professional annotators to ensure accuracy and consistency, making a very optimal dataset for training an object detection algorithm.


Back To Top

Solution Statement

Convolutional Neural Network (CNN) will be a perfect fit for this problem. In ordre to solve problems such as object detection, capturing spatial information is very crucial and CNN is a great fit. CNN's filters are able to capture spatial features (lines, curves, squares,..) and are able to detect objects using learned filters.


Back To Top

Benchmark Model

In the domain of object detection, there are many proven neural network architectures such as VGGNet, GoogLeNet, Resnet, DenseNet, and etc. There are widely available weights, pre-trained on large dataset similar to Open Images Dataset, from aforementioned neural network architectures. Using pre-trained weights from aforementioned architectures can be used to predict objects, and score from the base model will be used as a benchmark.


Back To Top

Evaluation Metrics

Benchmark model and the solution's performances will be evaluated by mean Average Precision (AP) (mAPmAP) across 500 classes in 1.7 million images in the dataset.

Precision (PP) will be calculated by dividing true positives (TpTp) over the true positives plus false positives (FpFp) (Precision-Recall)

P=TpTp+Fp
P=TpTp+Fp

Detections are considered true or false positives based on the area of overlap with ground truth bounding boxes. To be considered a correct detection, thearea of overlap aoao between the predicted bounding box BpBp and ground truth bounding box BgtBgt must exceed 50% by the formula (PASCAL VOC 2010):

ao>0.5 where ao=area(Bp∩Bgt)area(Bp∪Bgt)
ao>0.5 where ao=area(Bp∩Bgt)area(Bp∪Bgt)

Recall (RR) is calculated by dividing true positives (TpTp) over the true positives plus the false negatives (FnFn) (Precision-Recall)

R=TpTp+Fn
R=TpTp+Fn

Then, average precision (APAP) will be calculated for each of 500 classes.

AP=∑k=1n(Rk−Rk−1)Pk
AP=∑k=1n(Rk−Rk−1)Pk

Finally, final mean average precision (mAPmAP) willl be calculated by taking average of all APs over the 500 classes.
mAP=∑500k=1APk500
mAP=∑k=1500APk500

Note that, unlike previous standard challenges such as PASCAL VOC 2010, Google AI Open Images - Object Detection Track has three new metrics that affect the way True Positives and False Positives are accounted (Open Images Challenge 2018 - object detection track - evaluation metric)

    Due to the Open Images annotation process, image-level labeling is not exhaustive.
    The object classes are organized in a semantic hierarchy, meaning that some categories are more general than others (e.g. 'Animal' is more general than 'Cat', as 'Cat' is a subclass of 'Animal').
    Some of the ground-truth bounding-boxes capture a group of objects, rather than a single object.


Back To Top

Project Design

    Benchmark Model
        My first approach is to utilize pre-trained weights from most current Convolutional Neural Network (CNN) architectures such as VGG-19, ResNet-50, Inception, Xception, and etc. We will initially obtain benchmark model by evaluating which architecture's pre-trained weights give best mean average precision right off the bet.

    Improve Base Model
        Data will be analyzed if there are major impalances. If so, data augmentation techniques such as ADASYN, SMOTE, scaling, and etc can be used.
        Split dataset to training (~80%) and validation set (~20%) to be used to identify the most effective model. (For this competition, training and validation sets are provided)
        Then, model will tweaked and improved to yield the highest mean Average Precision (mAPmAP). Some of important categories for optimization potentials are learning rate, types of optimizers, usage of model ensembling, and so on.

    Final Touch
        Visualize file results in order to identify some of the best and worst performing classes. This will help us to find room for improvement in the model. One popular way is done by using a confusion matrix.
        Finally, train the model using entire dataset (training and validation set combined). Then, obtain predictions for testing image dataset and submit it in Kaggle.



Back To Top
