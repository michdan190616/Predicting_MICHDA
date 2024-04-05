# The MICHDA classification project

### Machine Learning course (CS-433)

## General information 

This repository contains the report and code for the first project of Machine Learning course (CS-433).

## Team

Our team is composed by:
- Andrea Belvisi
- Michelangelo D'Andrea
- Matteo Ferrazzi

The name of our team that you can find in AIcrowd is MMA

## Dataset

The datasets 'x_train.csv', 'y_train.csv' and 'x_test.csv' can be found in the ML course repository: https://github.com/epfml/ML_course.

Our submissions are stored in 'log_reg.csv', log_reg_reg.csv', 'ridge_reg.csv', 'mean_sq_gd.csv' and 'mean_sq_sgd.csv' respectively for logistic regression, regularized logistic regression, ridge regression, gradient descent, stochastic gradient descent.

## Code structure 

- 'helpers.py' is a provided file that contains useful functions for downloading the data and create the predictions.
- 'feat_eng.py' contains the code used to do data cleaning and feature engineering
- 'implementations.py' contains the code where we implemented the six methods
- 'model_selectors_predictors.py' contains the code used to do cross-validation
- 'run.ipynb' contains the code to run the cross validation and to obtain the submissions.

## Usage

You need to clone this repository.

Before running the code you will need to download the dataset 'x_train.csv', 'y_train.csv' and 'x_test.csv' from ML course repository: https://github.com/epfml/ML_course and store them in a folder called 'dataset_to_release'.

Finally you can run the 'run.ipynb' to get submissions.

## Results

Our best submissions used the logistic regression and obtained a F1 score of 0.421 in the 4-fold cross-validation and a F1 score of 0.427 on the test set.
Please note that AIcrowd still shows the #240899 submission as our best submission but, the latest version of our code produces the #242622 submission. Note that the two give exactly the same results in terms of both F1 score and accuracy, but AIcrowd kept the oldest one. We have done only small changes between the two submissions that do not affect the performance.
