#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:54:39 2021

@author: apoghosyan
"""

""" Module to read data """

import pandas as pd
import os
import glob

import pickle
import hydra
from hydra import utils


def read_data(file_path):

    """ Reads and cleans the data into a pandas dataframe. All csv files are stored in one folder.
    :param file_path: path to data
    :type file_path: str
    :param label_bool: whether the label is included in dataframe
    :type label_bool: bool
    :returns:
        - X - features dataframe
        - y - labels dataframe for cl_0
    """

    all_files = glob.glob(file_path + "/*.csv")
    li = []

    for filename in all_files:
        subdf = pd.read_csv(filename, index_col=None, header=0)
        li.append(subdf)

    df = pd.concat(li, axis=0, ignore_index=True)
    # preliminary analysis has shown 20 (2%) rows with target values undefined 'failed'. dropping those rows
    # convert target columns to numeric
    df = df.loc[((df['cl_0'] != 'failed') & (df['cd_0'] != 'failed') & (df['cm_0'] != 'failed'))]
    df['cl_0'] =pd.to_numeric(df['cl_0'])
    df['cd_0'] =pd.to_numeric(df['cd_0'])
    df['cm_0'] =pd.to_numeric(df['cm_0'])
    df=df.drop('ID_0', axis=1)
    x_df = df[df.columns[:-3]]
    y_df = df[df.columns[-3:]]
    return x_df, y_df['cl_0']

def save_file(dset, fname):
    print(utils.to_absolute_path(fname))
    dset.to_csv(utils.to_absolute_path(fname))
  #  with open(utils.to_absolute_path(name), "wb") as fp:
        #pickle.dump(file, fp)
