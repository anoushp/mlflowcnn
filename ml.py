#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:44:15 2021

@author: apoghosyan
"""

""" Module to train and evaluate ML model and to make predictions from trained models """

import pickle
import mlflow
import mlflow.sklearn
import xgboost
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from read_data import read_data
from read_data import save_file
from transform import Transformation


class Learner:

    """ Class to train & save the model """

    def __init__(self):
        self.folder_ml = '/Users/apoghosyan/Documents/daws2.4/mlflow-test/saved_models/'
        self.n_estimators = 5000
        self.model = None

    def train_model(self, X_train, y_train):

        """
        Training the model
        :param X_train: training features
        :type X_train: pandas dataframe
        :param y_train: training labels
        :type y_train: pandas dataframe
        """

        print('Fitting model...')
        self.model = XGBRegressor(booster='gbtree', 
                             max_depth = 4,
                    n_estimators=self.n_estimators, 
                    learning_rate = .02)

        self.model.fit(X_train, y_train)

    def save_model(self, model_name):

        """
        Saving the model using pickle
        :param model_name: name of model
        :type model_name: str
        """

        filename = model_name + '.pkl'
        pickle.dump(self.model, open(self.folder_ml + filename, 'wb'))

    def load_model(self, model_name):

        """
        Loading the model
        :param model_name: name of model
        :type model_name: str
        """

        filename = model_name + '.pkl'
        self.model = pickle.load(open(self.folder_ml + filename, 'rb'))


class Evaluation:

    """ Class to evaluate trained model """

    def __init__(self, mlflow_record):
        self.mlflow_record = mlflow_record
        self.validation_pc = 0.2
        self.X, self.y = read_data('/Users/apoghosyan/Documents/daws2.4/csvs/csv10')
        self.transform = Transformation()
        self.learner = Learner()
        self.model_name = 'model_eval'

    def mlflow_logging(self, model_metric):

        """
        Method to add data to mlflow
        :param model_metric: output from the model
        """

        # mlflow logging
        mlflow.log_param("validation_pc", self.validation_pc)
        mlflow.log_param("num_trees", self.learner.n_estimators)
        mlflow.log_metric("r2", model_metric)
        mlflow.sklearn.log_model(self.learner.model, "model")

    def split_train_test(self, X, y):

        """
        A method to split the features into a train and test set
        :param X: features
        :type X: pandas dataframe
        :param y: labels
        :type y: pandas dataframe
        :returns:
            - X_train - train features
            - X_test - test features
            - y_train - train labels
            - y_test - test labels
        """

        # split the dataframes
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=self.validation_pc, random_state=42)
        save_file(X_train, 'data/daws_train.csv')
        save_file(X_valid, 'data/daws_val.csv')
        save_file(y_train, 'data/daws_train_label.csv')
        save_file(y_valid, 'data/daws_val_label.csv')
     

        return X_train, X_valid, y_train, y_valid

    def train_and_evaluate(self, X_train, X_valid, y_train, y_valid):

        """
        Train and evaluate the performance of the model
        :param X_train: training features
        :type X_train: pandas dataframe
        :param X_valid: validation features
        :type X_valid: pandas dataframe
        :param y_train: training labels
        :type y_train: pandas dataframe
        :param y_valid: validation labels
        :type y_valid: pandas dataframe
        :returns:
            - metric - output model features
        """

        # training
        self.learner.train_model(X_train, y_train)
        self.learner.save_model(model_name=self.model_name)

        # evaluate
        print('Evaluating model...')
        y_valid_pred = self.learner.model.predict(X_valid)
        metric = r2_score(y_valid, y_valid_pred)

        return metric

    def run_single(self, exp_name):

        """
        Splitting, training, evaluating and logging model
        :param exp_name: name of experiment
        :type exp_name: str
        """

        # transform
        print('Tranforming data...')
        X_train, X_valid, y_train, y_valid = self.split_train_test(X=self.X, y=self.y)
        X_train, X_valid = self.transform.transform_data(X_train=X_train, X_test=X_valid)

        # train and evaluate
        if self.mlflow_record:
            mlflow.set_experiment(exp_name)
            with mlflow.start_run():
                metric = self.train_and_evaluate(X_train, X_valid, y_train, y_valid)
                self.mlflow_logging(model_metric=metric)
        else:
            metric = self.train_and_evaluate(X_train, X_valid, y_train, y_valid)

        print('R2 Score:', metric)

    def run_cv(self, exp_name):

        """
        Splitting, training, evaluating and logging model - cross validation
        :param exp_name: name of experiment
        :type exp_name: str
        """
        print("aaa")
        kf = KFold(n_splits=5, shuffle=True)
        metric_list = []
        count = 0

        # perform CV
        for train_index, valid_index in kf.split(self.X):

            # counter
            count += 1
            print('Run:', count)

            # train data
            y_train = self.y.iloc[train_index]
            y_valid = self.y.iloc[valid_index]
            X_train, X_valid = self.transform.transform_data(X_train=self.X.iloc[train_index, :], X_test=self.X.iloc[valid_index, :])

            # run data
            metric = self.train_and_evaluate(X_train, X_valid, y_train, y_valid)
            metric_list.append(metric)

        cv_metric = sum(metric_list) / len(metric_list)

        # track results
        if self.mlflow_record:
            mlflow.set_experiment(exp_name)
            with mlflow.start_run():
                self.mlflow_logging(model_metric=cv_metric)

        print('R2 Score:', cv_metric)


class Prediction:

    def __init__(self):
        self.X_train, self.y_train = read_data('data/train.csv', label_bool=True)
        self.X_test = read_data('data/test.csv', label_bool=False)
        self.transform = Transformation()
        self.learner = Learner()
        self.model_name = 'model_full'

    def train_model(self):

        """
        Training the model on training data
        """

        X_train, _ = self.transform.transform_data(X_train=self.X_train, X_test=self.X_train)
        self.learner.train_model(X_train, self.y_train)
        self.learner.save_model(model_name=self.model_name)

    def predict_test_values(self):

        """
        Making predictions on the test set
        :returns:
            - predictions - array of test set predictions
        """

        # transform data
        X_test, _ = self.transform.transform_data(self.X_test, self.X_test)

        # load model
        self.learner.load_model(model_name=self.model_name)

        # predict the probability of success
        predictions = self.learner.model.predict_proba(X_test)[:, -1][0]

        return predictions
