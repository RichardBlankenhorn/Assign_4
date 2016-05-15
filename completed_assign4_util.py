__author__ = 'Richard'
import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
import sys

# function for separating different kinds of variables
def variable_type(df, nominal_level = 3):
    categorical, numeric, nominal = [],[],[]
    for variable in df.columns.values:
        if np.issubdtype(np.array(df[variable]).dtype, int) or np.issubdtype(np.array(df[variable]).dtype, float):
            if len(np.unique(np.array(df[variable]))) <= nominal_level:
                nominal.append(variable)
            else:
                numeric.append(variable)
        else:
            categorical.append(variable)
    return numeric,categorical,nominal

# find variables with missing values
def variables_with_missing(df):
    result = []
    col_names = df.columns.tolist()
    for variable in col_names:
        percent = float(sum(df[variable].isnull()))/len(df.index)
        #print variable+":", percent
        if percent != 0:
            result.append(variable)
    return result

# draw histograms
def draw_histograms(df, variables, n_rows, n_cols):

    fig = plt.figure()
    length = len(variables)
    count = length - 1
    for item in variables:
        ax = fig.add_subplot(n_rows, n_cols, length-count)
        df[item].hist(bins=20, ax=ax)
        plt.title(item + ' distribution')
        count -= 1

    plt.show()

    return

# draw pie charts
def draw_piecharts(df, variables, n_rows, n_cols):

    length = len(variables)
    count = length - 1
    fig = plt.figure()
    for item in variables:
        ax = fig.add_subplot(n_rows, n_cols, length - count)
        df[item].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, title=item + ' distribution')
        count -= 1

    plt.show()

    return

# log transformation
def add_log_transform(df, variable, indexplus = 1):
    log_variable = "log_" + variable

    index = df.columns.get_loc(variable)

    log_num = np.log(df[variable]+1)
    df.insert(index+indexplus, log_variable, log_num)

    return

# split traing/test
def split_train_test_frame(df, test_size=.3):
    from sklearn.cross_validation import train_test_split
    df.reset_index(level=0, inplace=True)
    train, test = train_test_split(df.values, test_size=test_size)
    train_ = df.iloc[train[:,0].astype(int).tolist()]
    test_ = df.iloc[test[:,0].astype(int).tolist()]
    del train_['index']
    del test_['index']
    #train_.to_csv("train.csv")
    #test_.to_csv("test.csv")
    return (train_, test_)

def split_train_test_array(arr, test_size=.3):
    from sklearn.cross_validation import train_test_split
    train, test = train_test_split(arr.values, test_size=test_size)
    train_X = train[:, :-1]
    #print train_X.shape
    train_y = train[:, -1]
    #print train_y.shape
    test_X = test[:,:-1]
    test_y = test[:, -1]
    return (train_X,train_y,test_X,test_y)

def split_x_y(train):
    train_X = train[:, :-1]
    train_y = train[:, -1]
    return (train_X,train_y)

def process_missing_numeric_with_dummy(df, variable):
    missing_variable = variable + "_missing"
    df[missing_variable] = np.where(df[variable].isnull(),1,0)
    median = df[variable].median()
    df[variable].fillna(median, inplace= True)
    return df

def process_missing_numeric_no_dummy(df, variable):
    median = df[variable].median()
    df[variable].fillna(median, inplace= True)
    return df

def cap_variable(df, variable, num_std):
    upbound = df[variable].mean() + num_std * df[variable].std()
    df[variable] = df[variable].clip(upper = upbound)

