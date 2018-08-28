**Refer to "Capstone Project Proposal.ipynb" or "proposal.pdf" for details**

## [Data Preparation] Train/Validation/Test Image and Annotation Dataset

**Dataset Download :** [[Link1]](https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/) [[Link2]](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations) 

**Insights and Explanation on Dataset :** [Open Images Dataset v4](https://storage.googleapis.com/openimages/web/index.html)

## [Training Environement] Darknet Installation

**Original Darknet Architecture Developer :** [Joseph Redmon](https://pjreddie.com/) 

**Use This Link For Darknet Installation (More User Friendly) :** [AlexeyAB's Repository](https://github.com/AlexeyAB/darknet) - This is where you will find all of information about Darknet installation, commands to train & validate & test, and troubleshooting.

## [Actual Gears To Progress Through Project] Jupyter Notebooks
**Data Preparation for Darknet Training.ipynb :** Processes annotation files (.csv) provided by Kaggle competition to be used for Darknet training.  

**Train Data Analysis & Fix Data Imbalance.ipynb :** Provides an analysis on training dataset's imbalance. It also addresses data imbalance.

**Merge Darknet Output (Ensemble Submission).ipynb :** Merges predictions made on same dataset by different models using Darknet. Takes output from Darknet in .txt format, merges predictions, and converts it to .csv file required for Kaggle competition submission here : [Submit](https://www.kaggle.com/c/google-ai-open-images-object-detection-track/submit)

## [Etc] Relevant Files
**iter graph.xlsx :** Kept track of mAP on validation set versus iteration for the two models (model to predict entire 500 classes and model to predict only 420 classes excluding 80 classes covered by COCO model).

**detector.c :** I made a modification in detector.c from AlexeyAB's repository to have Darknet output predictions in the format Kaggle competition will take. Therefore, use this file instead of original detector.c from the repo.

**image.c :** I made a modification in image.c from AlexeyAB's repository to have Darknet output predictions in the format Kaggle competition will take. Therefore, use this file instead of original image.c from the repo.
