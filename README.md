# ICU Mortality Time-Series Prediction

## Overview
This project aims to predict ICU patient mortality using time-series data from the [PhysioNet 2012 Challenge](https://physionet.org/content/challenge-2012/1.0.0/) dataset. The dataset consists of 37 physiological variables recorded over 48 hours, along with static patient information. The objective is to develop machine learning models to assist clinical decision-making.

## How to run this code to get all results from A-Z

Create a new environment using the requirements or use the student-cluster and make sure to add optuna

### 1_EDA

* make sure to get the data on the studen cluster in ml4h/p1 and match it to the path given in 01-data-processing-exploration.ipynb
* Run 01-data-processing-exploration.ipynb to do the preprocessing and get the parquets in ./data

### 2_SupervisedML

* @damla in what order goes what here?
* Run Q2_RNN/2_RNN.ipynb to get the results for the LSTM and BiLSTM
* Run 02-3-transformer-(tokens) to obtain the results for the transformer task in 2

### 3_RepresentationLearning

* Run 3.1_Encoder to get the results of the LSTM encoder
* Run 3.2_LabelScarcity_Complete_FINAL to do the scarcity experiments and get the patient_embeddings_* for 3.3

### 4_FoundationModels

* Uncomment all cells in Q4.1_LLM4TS_problem.ipynb to get the embeddings and run the notebook the get the results stated in the report
* Visualize the embeddings with 04-3-embeddings.visualizations.ipynb
* Uncomment all cells and run to get the Chronos results for q4.3. make sure to have 17GB of disk space, otherwise *.pt file cannot be written to disk (needed for q4.3.2)

