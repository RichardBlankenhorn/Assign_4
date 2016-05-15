__author__ = 'Richard'
import pandas as pd
import numpy as np
import completed_assign4_util as util
import sys
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn import grid_search

def read_data():
    df = pd.read_csv('credit.csv')
    assert isinstance(df, pd.DataFrame) # for pycharm code completion
    #print df.head()
    #print df.describe()
    # remove duplicates
    df = df.drop_duplicates()
    # remove rows with dependent variable missing
    df = df.dropna(subset=['TARGET'])
    return df

def missing_imputation_for_numeric(df, numeric_with_na):
    for var in numeric_with_na:
        if "log_" in var or "sqrt_" in var:
            util.process_missing_numeric_no_dummy(df, var)
        else:
            util.process_missing_numeric_with_dummy(df, var)
    return

def variable_selection(train, test, model):
    train_x, train_y = util.split_x_y(train.values)
    #test_x, test_y = util.split_x_y(test.values)

    train_vars = train.columns.tolist()

    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(train_y, 5),
            scoring='accuracy')

    selector = rfecv.fit(train_x, train_y)
    print selector.support_

    new_train_list = []
    count = 0
    for value in selector.support_:
        if value == True:
            new_train_list.append(train_vars[count])
            count += 1
    new_train_list.append('TARGET')

    return new_train_list

# the method below creates a new training and a new test data set that include only the selected variables
def train_test_keep_some_vars(train, test, variables):
    train_new = train[variables]
    test_new = test[variables]
    return (train_new, test_new)

def fit_logistic_regression(train_new, test_new):
    train_x, train_y = util.split_x_y(train_new.values)
    test_x, test_y = util.split_x_y(test_new.values)

    num_folds = 5
    num_instances = len(train_x)
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds)
    param_grid = [{'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
    model = grid_search.GridSearchCV(LogisticRegression(penalty='l2', class_weight='balanced'), param_grid, cv=kfold, scoring='accuracy')
    model.fit(train_x, train_y)

    print "Best parameters set found on development set:"
    print model.best_estimator_
    predicted = model.predict(test_x)
    expected = test_y
    print "The logistic regression classification results:"
    print(metrics.classification_report(expected, predicted))
    #print(metrics.confusion_matrix(expected, predicted))

def fit_svc(train_new, test_new):
    train_x, train_y = util.split_x_y(train_new.values)
    test_x, test_y = util.split_x_y(test_new.values)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)

    # fit a SVM model to the data
    param_grid = [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}]
    num_folds = 5
    num_instances = len(train_x)
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds)
    model = grid_search.GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=kfold, scoring='accuracy')

    model.fit(train_x, train_y)

    print "Best parameters set found on development set:"
    print model.best_estimator_
    predicted = model.predict(test_x)
    # summarize the fit of the model
    print "The SVC classification results:"
    print(metrics.classification_report(train_y, predicted))
    #print(metrics.confusion_matrix(train_y, predicted))

def main():
    # Step 1. Import data
    df = read_data()
    # Step 2. Explore data
    # 2.1. Get variable names
    col_names = df.columns.tolist()
    # 2.2. Classify variables into numeric, categorical (with strings), and nominal
    numeric,categorical,nominal = util.variable_type(df)
    print "numeric:", numeric # ['ID', 'DerogCnt', 'CollectCnt', 'InqCnt06', 'InqTimeLast', 'InqFinanceCnt24', 'TLTimeFirst', 'TLTimeLast', 'TLCnt03', 'TLCnt12', 'TLCnt24', 'TLCnt', 'TLSum', 'TLMaxSum', 'TLSatCnt', 'TLDel60Cnt', 'TLBadCnt24', 'TL75UtilCnt', 'TL50UtilCnt', 'TLBalHCPct', 'TLSatPct', 'TLDel3060Cnt24', 'TLDel90Cnt24', 'TLDel60CntAll', 'TLOpenPct', 'TLBadDerogCnt', 'TLDel60Cnt24', 'TLOpen24Pct']
    print "categorical:", categorical # no categorical
    print "nominal:", nominal # ['TARGET', 'BanruptcyInd']
    # 2.3. Draw histogram for numeric variables
    util.draw_histograms(df, ['DerogCnt', 'CollectCnt', 'InqCnt06', 'InqTimeLast', 'InqFinanceCnt24', 'TLTimeFirst', 'TLTimeLast', 'TLCnt03', 'TLCnt12'], 3,3)
    util.draw_histograms(df, ['TLCnt24', 'TLCnt', 'TLSum', 'TLMaxSum', 'TLSatCnt', 'TLDel60Cnt', 'TLBadCnt24', 'TL75UtilCnt', 'TL50UtilCnt' ], 3,3)
    util.draw_histograms(df, ['TLBalHCPct', 'TLSatPct', 'TLDel3060Cnt24', 'TLDel90Cnt24', 'TLDel60CntAll', 'TLOpenPct', 'TLBadDerogCnt', 'TLDel60Cnt24', 'TLOpen24Pct'], 3,3)
    # 2.4. Identify variables that have skewed distribution and need to be log or sqrt-transformed
    variables_needs_tranform = ['DerogCnt', 'CollectCnt', 'InqCnt06', 'InqTimeLast', 'InqFinanceCnt24', 'TLTimeFirst', 'TLTimeLast', 'TLCnt03', 'TLCnt12', 'TLCnt24', 'TLCnt', 'TLSum', 'TLMaxSum', 'TLDel60Cnt', 'TLBadCnt24', 'TL75UtilCnt', 'TL50UtilCnt', 'TLBalHCPct', 'TLSatPct', 'TLDel3060Cnt24', 'TLDel90Cnt24', 'TLDel60CntAll', 'TLBadDerogCnt', 'TLDel60Cnt24', 'TLOpen24Pct']
    # 2.5. Draw pie charts for categorical variables
    util.draw_piecharts(df, [ 'TARGET', 'BanruptcyInd'], 1,2)
    # Step 3. Transform variables
    for value in variables_needs_tranform:
        util.add_log_transform(df, value)
    #print df.head()
    # 3.3 Missing value imputation
    numeric,categorical,nominal = util.variable_type(df)
    variables_with_na = util.variables_with_missing(df)
    #print numeric
    print variables_with_na
    numeric_with_na = []
    nominal_with_na = []
    for val in numeric:
        if val in variables_with_na:
            numeric_with_na.append(val)

    print numeric_with_na
    print nominal_with_na
    missing_imputation_for_numeric(df, numeric_with_na) # do missing value imputation
    print df.head()

    # after transformation and missing value imputation, put the target variable as the last column
    vars = ['DerogCnt', 'log_DerogCnt',  'CollectCnt', 'log_CollectCnt',  'BanruptcyInd', 'InqCnt06', 'log_InqCnt06', 'InqTimeLast', 'log_InqTimeLast',  'InqFinanceCnt24', 'log_InqFinanceCnt24',  'TLTimeFirst', 'log_TLTimeFirst', 'TLTimeLast', 'log_TLTimeLast',  'TLCnt03', 'log_TLCnt03',  'TLCnt12', 'log_TLCnt12',  'TLCnt24', 'log_TLCnt24', 'TLCnt', 'log_TLCnt',  'TLSum', 'log_TLSum',  'TLMaxSum', 'log_TLMaxSum', 'TLSatCnt', 'TLDel60Cnt', 'log_TLDel60Cnt',  'TLBadCnt24', 'log_TLBadCnt24',  'TL75UtilCnt', 'log_TL75UtilCnt', 'TL50UtilCnt', 'log_TL50UtilCnt',  'TLBalHCPct', 'log_TLBalHCPct',  'TLSatPct', 'log_TLSatPct', 'TLDel3060Cnt24', 'log_TLDel3060Cnt24',  'TLDel90Cnt24', 'log_TLDel90Cnt24',  'TLDel60CntAll', 'log_TLDel60CntAll', 'TLOpenPct', 'TLBadDerogCnt', 'log_TLBadDerogCnt',  'TLDel60Cnt24', 'log_TLDel60Cnt24', 'TLOpen24Pct', 'log_TLOpen24Pct', 'TLMaxSum_missing', 'TL50UtilCnt_missing', 'TLOpenPct_missing', 'TLBalHCPct_missing', 'TLSum_missing', 'TL75UtilCnt_missing', 'TLSatCnt_missing', 'TLCnt_missing', 'TLSatPct_missing', 'TLOpen24Pct_missing', 'InqTimeLast_missing', 'TARGET']

    df = df[vars]
    # Step 4. Split data into training and test
    train, test = util.split_train_test_frame(df, test_size=.5)
    # Step 5. Variable selection using a logistic regression model.
    model = LogisticRegression()
    variable_selection(train, test, model)
    selected_vars = variable_selection(train, test, model)
    #print selected_vars
    # Step 6. Model fitting and evaluation.
    train_new, test_new= train_test_keep_some_vars(train, test, selected_vars)
    #print train_new
    fit_logistic_regression(train_new, test_new)
    fit_svc(train_new, test_new)

if __name__ == "__main__":
    main()










