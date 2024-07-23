#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:40:29 2021

@author: apoghosyan
"""

""" Module to run src files """

from ml import Evaluation, Prediction


def run():

    """
    Function to run the machine learning model
    """

    # evaluate model with single split
    eval_model = Evaluation(mlflow_record=True)
    eval_model.split_train_test(X=eval_model.X, y=eval_model.y)
    #eval_model.run_single(exp_name='single_run')

    # # evaluate model with cross-validation
    # eval_model = Evaluation(mlflow_record=False)
    
    #eval_model.run_cv(exp_name='cross_validation')

    # # saved trained model on whole dataset
    # predict = Prediction()
    # predict.train_model()


if __name__ == '__main__':

    run()
