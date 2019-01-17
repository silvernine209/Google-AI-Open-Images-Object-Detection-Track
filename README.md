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








*Instructions: Click on the raw button in the upper right hand corner of this box.  Copy and paste the template into the README.md document on your github.  Fill in the titles, information and links where prompted! Feel free to stray a bit to suit your project but try to stick to the format as closely as possible for consistency across DSWG projects.*

# Project Name
This project is a part of the [Data Science Working Group](http://datascience.codeforsanfrancisco.org) at [Code for San Francisco](http://www.codeforsanfrancisco.org).  Other DSWG projects can be found at the [main GitHub repo](https://github.com/sfbrigade/data-science-wg).

#### -- Project Status: [Active, On-Hold, Completed]

## Project Intro/Objective
The purpose of this project is ________. (Describe the main goals of the project and potential civic impact. Limit to a short paragraph, 3-6 Sentences)

### Partner
* [Name of Partner organization/Government department etc..]
* Website for partner
* Partner contact: [Name of Contact], [slack handle of contact if any]
* If you do not have a partner leave this section out

### Methods Used
* Inferential Statistics
* Machine Learning
* Data Visualization
* Predictive Modeling
* etc.

### Technologies
* R 
* Python
* D3
* PostGres, MySql
* Pandas, jupyter
* HTML
* JavaScript
* etc. 

## Project Description
(Provide more detailed overview of the project.  Talk a bit about your data sources and what questions and hypothesis you are exploring. What specific data analysis/visualization and modelling work are you using to solve the problem? What blockers and challenges are you facing?  Feel free to number or bullet point things here)

## Needs of this project

- frontend developers
- data exploration/descriptive statistics
- data processing/cleaning
- statistical modeling
- writeup/reporting
- etc. (be as specific as possible)

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [here](Repo folder containing raw data) within this repo.

    *If using offline data mention that and how they may obtain the data from the froup)*
    
3. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
4. etc...

*If your project is well underway and setup is fairly complicated (ie. requires installation of many packages) create another "setup.md" file and link to it here*  

5. Follow setup [instructions](Link to file)

## Featured Notebooks/Analysis/Deliverables
* [Notebook/Markdown/Slide Deck Title](link)
* [Notebook/Markdown/Slide DeckTitle](link)
* [Blog Post](link)
