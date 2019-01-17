# Google AI Open Images - Object Detection Track
This repository captures my efforts to compete in the Kaggle competition:[Google AI Open Images - Object Detection Track](https://www.kaggle.com/c/google-ai-open-images-object-detection-track) by training a CNN.

#### -- Project Status: [Completed]

## Project Intro/Objective
Goal of the competition was to build an algorithm that detects objects using 1.7 million image dataset.
There were total of 500 classes of objects which were thoroughly annotated in the given dataset.

### Partner
* [Kaggle](https://www.kaggle.com/c/google-ai-open-images-object-detection-track)
* [Open Images Dataset V4](https://storage.googleapis.com/openimages/web/download.html)

### Methods Used
* Convolutional Neural Network (CNN)
* Data Visualization
* Data Pre-Processing
* etc.

### Technologies
* Python
* Jupyter Notebook
* Pandas, Numpy, PIL, Matplotlib
* Darknet
* YoloV3
* etc. 

### TechnologiesObtaining Dataset
* **Dataset Download :** [[Link1]](https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/) [[Link2]](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations)  
* **Insights and Explanation on Dataset :** [Open Images Dataset v4](https://storage.googleapis.com/openimages/web/index.html)

### Configuring Darknet Training Environment
* **Original Darknet Architecture Developer :** [Joseph Redmon](https://pjreddie.com/)  
* **Use This Link For Darknet Installation (More User Friendly) :** [AlexeyAB's Repository](https://github.com/AlexeyAB/darknet) - This is where you will find all of information about Darknet installation, commands to train & validate & test, and troubleshooting.

### Jupyter Notebooks Used
**Data Preparation for Darknet Training.ipynb :** Processes annotation files (.csv) provided by Kaggle competition to be used for Darknet training.  

**Train Data Analysis & Fix Data Imbalance.ipynb :** Provides an analysis on training dataset's imbalance. It also addresses data imbalance.

**Merge Darknet Output (Ensemble Submission).ipynb :** Merges predictions made on same dataset by different models using Darknet. Takes output from Darknet in .txt format, merges predictions, and converts it to .csv file required for Kaggle competition submission here : [Submit](https://www.kaggle.com/c/google-ai-open-images-object-detection-track/submit)

### Relevant Files Needed to Replicate the Work
**iter graph.xlsx :** Kept track of mAP on validation set versus iteration for the two models (model to predict entire 500 classes and model to predict only 420 classes excluding 80 classes covered by COCO model).

**detector.c :** I made a modification in detector.c from AlexeyAB's repository to have Darknet output predictions in the format Kaggle competition will take. Therefore, use this file instead of original detector.c from the repo.

**image.c :** I made a modification in image.c from AlexeyAB's repository to have Darknet output predictions in the format Kaggle competition will take. Therefore, use this file instead of original image.c from the repo.


## Featured Notebooks/Analysis/Deliverables
* [Project Proposal](https://github.com/silvernine209/Google-AI-Open-Images-Object-Detection-Track/blob/master/proposal.pdf)
* [Exploratory Data Analysis on Training Dataset](https://github.com/silvernine209/Google-AI-Open-Images-Object-Detection-Track/blob/master/Jupyter%20Notebooks/Train%20Data%20Analysis%20%26%20Fix%20Data%20Imbalance.ipynb)
* [Data Preparation for Darknet Training](https://github.com/silvernine209/Google-AI-Open-Images-Object-Detection-Track/blob/master/Jupyter%20Notebooks/Data%20Preparation%20for%20Darknet%20Training.ipynb)
* [Project Final Report](https://github.com/silvernine209/Google-AI-Open-Images-Object-Detection-Track/blob/master/Project%20Final%20Report.pdf)

