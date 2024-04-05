import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from implementations import *
from feat_eng import *
from helpers import *

#############------------------------------##############

def accuracy(y, y_pred):
    """
    A function that takes as inputs:
    - y: the array of dependent variable observations
    - y: the predictions of the model

    And returns the accuracy of the model.

    """
    return np.mean(y == y_pred)

#############------------------------------#############

def F1_score(y, y_hat):
    """
    A function that takes as inputs:
    - y: the array of dependent variable observations
    - y: the predictions of the model

    And returns the F1 score of the model.

    """
    return 2 * np.sum(y * y_hat) / (np.sum(y) + np.sum(y_hat))

#############------------------------------#############

def k_fold_cross_val_reg_logistic_regression(y,datas,rates,lambdas,max_iters,k, thresholds):
    """
    A function that performs the k-fold validation of the Regularized Logistic Regression and tunes
    4 hyperparameters:
    - datas: the lists of dataset with the different polynomial features, represent the tuning of the degree
    - thresholds: a list of different decision thresholds
    - rates: a list of learning rates
    - lambdas: a list of different regularization terms

    The function returns two dictionaries:
    - accuracy_dict: contains the accuracy of the model for each combination of the hyperparameters
    - F1_score_dict: contains the F1 of the model for each combination of the hyperparameters

    """
    accuracy_dict = {}
    F1_score_dict = {}
    for j,df in enumerate(datas):
        xx_train,yy_train,xx_test,yy_test = split_dataset(df,y,k)
        initial_w = np.zeros(xx_train[0].shape[1])
        for rate in rates:
            for l in lambdas:
                acc_arr = np.zeros(len(thresholds))
                F1_arr = np.zeros(len(thresholds))

                for i in range(k):

                    w, loss = reg_logistic_regression(yy_train[i], xx_train[i], l ,initial_w, max_iters, rate)

                    for idx,t in enumerate(thresholds):

                        acc_arr[idx]+=accuracy(yy_test[i], np.array(logistic(xx_test[i]@w)>=t,dtype=int))
                        F1_arr[idx]+=F1_score(yy_test[i], np.array(logistic(xx_test[i]@w)>=t,dtype=int))

                for idx,t in enumerate(thresholds):
                    accuracy_dict[(j+1, 'reg_logistic_regression', rate, l, t)] = acc_arr[idx]/k
                    F1_score_dict[(j+1, 'reg_logistic_regression', rate, l, t)] = F1_arr[idx]/k              

    return accuracy_dict, F1_score_dict

#############------------------------------#############

def k_fold_cross_val_ridge_regression(y,datas,lambdas,thresholds,k):
    """
    A function that performs the k-fold validation of the Ridge Regression and tunes
    3 hyperparameters:
    - datas: the lists of dataset with the different polynomial features, represent the tuning of the degree
    - thresholds: a list of different decision thresholds
    - lambdas: a list of different regularization terms

    The function returns two dictionaries:
    - accuracy_dict: contains the accuracy of the model for each combination of the hyperparameters
    - F1_score_dict: contains the F1 of the model for each combination of the hyperparameters

    """
    accuracy_dict = {}
    F1_score_dict = {}
    for j,df in enumerate(datas):
        xx_train,yy_train,xx_test,yy_test = split_dataset(df,y,k)
        for l in lambdas:
            for t in thresholds:
                acc=0
                F1=0
                for i in range(k):
                    w, loss = ridge_regression(yy_train[i], xx_train[i], l)
                    acc+=accuracy(yy_test[i], np.array((xx_test[i]@w)>=t,dtype=int))
                    F1+=F1_score(yy_test[i], np.array((xx_test[i]@w)>=t,dtype=int))
                accuracy_dict[(j+1, 'ridge_regression', 0, l, t)] = acc/k
                F1_score_dict[(j+1, 'ridge_regression', 0, l, t)] = F1/k
                    
    return accuracy_dict, F1_score_dict

#############------------------------------#############

def k_fold_cross_val_mean_squared_error_gd(y,datas,rates,thresholds,max_iters,k):
    """
    A function that performs the k-fold validation of the Gradient Descent method and tunes
    3 hyperparameters:
    - datas: the lists of dataset with the different polynomial features, represent the tuning of the degree
    - thresholds: a list of different decision thresholds
    - rates: a list of learning rates

    The function returns two dictionaries:
    - accuracy_dict: contains the accuracy of the model for each combination of the hyperparameters
    - F1_score_dict: contains the F1 of the model for each combination of the hyperparameters

    """
    accuracy_dict = {}
    F1_score_dict = {}
    for j,df in enumerate(datas):
        xx_train,yy_train,xx_test,yy_test = split_dataset(df,y,k)
        initial_w = np.zeros(xx_train[0].shape[1])
        for rate in rates:
            for t in thresholds:
                acc=0
                F1=0
                for i in range(k):
                    w, loss = mean_squared_error_gd(yy_train[i], xx_train[i], initial_w, max_iters, rate/(j+1))
                    acc+=accuracy(yy_test[i], np.array((xx_test[i]@w)>=t,dtype=int))
                    F1+=F1_score(yy_test[i], np.array((xx_test[i]@w)>=t,dtype=int))
                accuracy_dict[(j+1, 'mean_squared_error_gd', rate, 0, t)] = acc/k
                F1_score_dict[(j+1, 'mean_squared_error_gd', rate, 0, t)] = F1/k

    return accuracy_dict, F1_score_dict

#############------------------------------#############

def k_fold_cross_val_mean_squared_error_sgd(y,datas,rates,thresholds,max_iters,k):
    """
    A function that performs the k-fold validation of the Stochastic Gradient Descent methos and tunes
    3 hyperparameters:
    - datas: the lists of dataset with the different polynomial features, represent the tuning of the degree
    - thresholds: a list of different decision thresholds
    - rates: a list of learning rates

    The function returns two dictionaries:
    - accuracy_dict: contains the accuracy of the model for each combination of the hyperparameters
    - F1_score_dict: contains the F1 of the model for each combination of the hyperparameters

    """
    accuracy_dict = {}
    F1_score_dict = {}
    for j,df in enumerate(datas):
        xx_train,yy_train,xx_test,yy_test = split_dataset(df,y,k)
        initial_w = np.zeros(xx_train[0].shape[1])
        for rate in rates:
            for t in thresholds:
                acc=0
                F1=0
                for i in range(k):
                    w, loss = mean_squared_error_sgd(yy_train[i], xx_train[i], initial_w, max_iters, rate/(j+1))
                    acc+=accuracy(yy_test[i], np.array((xx_test[i]@w)>=t,dtype=int))
                    F1+=F1_score(yy_test[i], np.array((xx_test[i]@w)>=t,dtype=int))
                accuracy_dict[(j+1, 'mean_squared_error_sgd', rate, 0, t)] = acc/k
                F1_score_dict[(j+1, 'mean_squared_error_sgd', rate, 0, t)] = F1/k

    return accuracy_dict, F1_score_dict

#############------------------------------#############

def k_fold_cross_val_logistic_regression(y,datas,rates,max_iters,k, thresholds):
    """
    A function that performs the k-fold validation of the Logistic Regression and tunes
    3 hyperparameters:
    - datas: the lists of dataset with the different polynomial features, represent the tuning of the degree
    - thresholds: a list of different decision thresholds
    - rates: a list of learning rates

    The function returns two dictionaries:
    - accuracy_dict: contains the accuracy of the model for each combination of the hyperparameters
    - F1_score_dict: contains the F1 of the model for each combination of the hyperparameters

    """
    accuracy_dict = {}
    F1_score_dict = {}
    for j,df in enumerate(datas):
        xx_train,yy_train,xx_test,yy_test = split_dataset(df,y,k)
        initial_w = np.zeros(xx_train[0].shape[1])
        for rate in rates:

            acc_arr = np.zeros(len(thresholds))
            F1_arr = np.zeros(len(thresholds))

            for i in range(k):

                w, loss = logistic_regression(yy_train[i], xx_train[i], initial_w, max_iters, rate)

                for idx,t in enumerate(thresholds):

                    acc_arr[idx]+=accuracy(yy_test[i], np.array(logistic(xx_test[i]@w)>=t,dtype=int))
                    F1_arr[idx]+=F1_score(yy_test[i], np.array(logistic(xx_test[i]@w)>=t,dtype=int))

            for idx,t in enumerate(thresholds):
                accuracy_dict[(j+1, 'logistic_regression', rate, 0, t)] = acc_arr[idx]/k
                F1_score_dict[(j+1, 'logistic_regression', rate, 0, t)] = F1_arr[idx]/k 

    return accuracy_dict, F1_score_dict

#############------------------------------#############

def y_build_ridge_reg(datas, y_train, x_test, F1_dict, names,sq_root):
    """
    A function that selects the best combination of hyperparameters in terms of F1 score for
    the Ridge Regression, and returns the best value of F1 score and relative prediction.

    """
    # select the best model based on the F1 score 
    best_model = max(F1_dict, key = lambda x: F1_dict[x])
    w, loss = ridge_regression(y_train, datas[best_model[0]-1], best_model[3])
    x_test = feat_engineering(x_test,names,best_model[0],sq_root)
    y_pred = np.array((x_test@w)>=best_model[4],dtype=int)
    y_pred = y_pred*2 - 1
    return y_pred, max(F1_dict.values())

#############------------------------------#############

def y_build_log_reg(datas, y_train, x_test, F1_dict, max_iters, names,sq_root):
    """
    A function that selects the best combination of hyperparameters in terms of F1 score for
    the Logistic Regression, and returns the best value of F1 score and relative prediction.

    """
    # select the best model based on the F1 score 
    best_model = max(F1_dict, key = lambda x: F1_dict[x])
    initial_w = np.zeros(datas[best_model[0]-1].shape[1])
    w, loss = logistic_regression(y_train, datas[best_model[0]-1], initial_w, max_iters, best_model[2])
    x_test = feat_engineering(x_test,names,best_model[0],sq_root)
    y_pred = np.array(logistic(x_test@w)>=best_model[4],dtype=int)
    y_pred = y_pred*2 - 1
    return y_pred, max(F1_dict.values())

#############------------------------------#############

def y_build_log_reg_reg(datas, y_train, x_test, F1_dict, max_iters, names,sq_root):
    """
    A function that selects the best combination of hyperparameters in terms of F1 score for
    the Regularized Logistic Regression, and returns the best value of F1 score and relative prediction.

    """
    # select the best model based on the F1 score 
    best_model = max(F1_dict, key = lambda x: F1_dict[x])
    initial_w = np.zeros(datas[best_model[0]-1].shape[1])
    w, loss = reg_logistic_regression(y_train, datas[best_model[0]-1], initial_w, max_iters, best_model[2], best_model[3])
    x_test = feat_engineering(x_test,names,best_model[0],sq_root)
    y_pred = np.array(logistic(x_test@w)>=best_model[4],dtype=int)
    y_pred = y_pred*2 - 1
    return y_pred, max(F1_dict.values())

#############------------------------------#############

def y_build_mean_sq_gd(datas, y_train, x_test, F1_dict, max_iters, names,sq_root):
    """
    A function that selects the best combination of hyperparameters in terms of F1 score for
    the Gradient Descent method, and returns the best value of F1 score and relative prediction.

    """
    # select the best model based on the F1 score 
    best_model = max(F1_dict, key = lambda x: F1_dict[x])
    initial_w = np.zeros(datas[best_model[0]-1].shape[1])
    w, loss = mean_squared_error_gd(y_train, datas[best_model[0]-1], initial_w, max_iters, best_model[2])
    x_test = feat_engineering(x_test,names,best_model[0],sq_root)
    y_pred = np.array((x_test@w)>=best_model[4],dtype=int)
    y_pred = y_pred*2 - 1
    return y_pred, max(F1_dict.values())

#############------------------------------#############

def y_build_mean_sq_sgd(datas, y_train, x_test, F1_dict, max_iters, names,sq_root):
    """
    A function that selects the best combination of hyperparameters in terms of F1 score for
    the Stochastic Gradient Descent method, and returns the best value of F1 score and relative prediction.

    """
    # select the best model based on the F1 score 
    best_model = max(F1_dict, key = lambda x: F1_dict[x])
    initial_w = np.zeros(datas[best_model[0]-1].shape[1])
    w, loss = mean_squared_error_sgd(y_train, datas[best_model[0]-1], initial_w, max_iters, best_model[2])
    x_test = feat_engineering(x_test,names,best_model[0],sq_root)
    y_pred = np.array((x_test@w)>=best_model[4],dtype=int)
    y_pred = y_pred*2 - 1
    return y_pred, max(F1_dict.values())
