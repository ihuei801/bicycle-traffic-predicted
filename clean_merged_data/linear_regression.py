import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
import csv
import time
import os
import math
import calendar
from datetime import datetime
from math import log
import plotly.plotly as py
import plotly.tools as tls

def convert_type(df):
    df['date'] = pd.to_datetime(df['date'], format="%m/%d/%Y")
    return df


def calculate_mse(x, y):
    return ((x - y) ** 2).mean()

def create_features(df):
    feature_df = pd.DataFrame()
    feature_df['count'] = df['count']
    feature_df['weather_type'] = df['weather_type']
    # normalize
    feature_df['temperature'] = df['temperature'] / max(df['temperature'])
    feature_df['humidity'] = df['humidity'] / max(df['humidity'])
    feature_df['wind_speed'] = df['wind_speed'] / max(df['wind_speed'])
    # circlic
    feature_df['sin_day_of_year'] = df['date'].map(lambda time: math.sin(
        2 * math.pi * datetime.date(time.to_datetime()).timetuple().tm_yday / 366) if calendar.isleap(
        time.to_datetime().year) else math.sin(
        2 * math.pi * datetime.date(time.to_datetime()).timetuple().tm_yday / 365))
    feature_df['cos_day_of_year'] = df['date'].map(lambda time: math.cos(
        2 * math.pi * datetime.date(time.to_datetime()).timetuple().tm_yday / 366.0) if calendar.isleap(
        time.to_datetime().year) else math.sin(
        2 * math.pi * datetime.date(time.to_datetime()).timetuple().tm_yday / 365.0))
    feature_df['sin_day_of_week'] = df['date'].map(
        lambda time: math.sin(2 * math.pi * time.to_datetime().weekday() / 7.0))
    feature_df['cos_day_of_week'] = df['date'].map(
        lambda time: math.cos(2 * math.pi * time.to_datetime().weekday() / 7.0))
    feature_df['sin_hour_of_day'] = df['hour'].map(
        lambda time: math.sin(2 * math.pi * time / 24.0))
    feature_df['cos_hour_of_day'] = df['hour'].map(
        lambda time: math.cos(2 * math.pi * time / 24.0))

    feature_df['weekday'] = df['date'].map(
        lambda time: 1 if time.to_datetime().weekday() <= 4 else 0)

    return feature_df


def extract_2011(df):
    df_2011 = df.loc[(df['date'] >= '2011-1-1') & (df['date'] < '2012-1-1')].reset_index(drop=True)
    return df_2011


def extract_2012(df):
    df_2012 = df.loc[(df['date'] >= '2012-1-1') & (df['date'] < '2013-1-1')].reset_index(drop=True)
    return df_2012


def extract_train(df):  # 2012-2013
    df_train = df.loc[(df['date'] >= '2012-1-1') & (df['date'] < '2014-1-1')].reset_index(drop=True)
    return df_train


def extract_test(df):  # 2014
    df_test = df.loc[(df['date'] >= '2014-1-1') & (df['date'] < '2015-1-1')].reset_index(drop=True)
    return df_test


def split_train_test(df):
    df_train = df.loc[df['date'] < '2015-1-1'].reset_index(drop=True)
    df_test = df.loc[df['date'] >= '2015-1-1'].reset_index(drop=True)
    return df_train, df_test

svm_res = []
dec_res = []
path = "./data_by_cluster/"
for filename in os.listdir(path):
    df = pd.read_csv(path+filename)
    # print df.head()
    # convert data to correct type:
    df = convert_type(df)
    # print df.head()

    df_train = extract_train(df)
    df_test = extract_test(df)
    # print df_train.head()
    # print len(df_train)
    # print df_test.head()
    # print len(df_test)
    df_train_feature = create_features(df_train)
    # print df_train_feature.head()
    df_test_feature = create_features(df_test)
    X_train = df_train_feature.drop('count', 1).values
    # print X_train[:10]
    y_train = df_train_feature['count'].values
    # print y_train[:10]

    # SVM
    print "SVM"
    # eps = np.arange(9,10.1,0.1).tolist()
    # tols = np.arange(2,3.1,0.1).tolist()
    # gamma = np.arange(0.5,1.1,0.1).tolist()
    #eps = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    svr = svm.SVR(C=100, gamma=0.7, epsilon=6.4)
    #print svr.get_params()
    clf = svr.fit(X_train, y_train)
    X_test = df_test_feature.drop('count', 1).values
    # print X_test[:10]
    y_test = df_test_feature['count'].values




    y_pred = clf.predict(X_test)
    y_pred = [max(0,x) for x in y_pred]
    x_sample = range(len(y_pred))

    #print x_sample[:10]
    #print y_pred[:10]


    # with open('svm_result.csv', 'wb') as result_file:
    #     csv_writer = csv.writer(result_file)
    #     for i in range(len(y_pred)):
    #         csv_writer.writerow([y_pred[i]])
    res = calculate_mse(y_pred, y_test)
    svm_res.append(res)

    #bike predictor
    print "bike predictor"
    Bestset = [19, 3, 2]
    Bestadaboost = [75, 0.05]
    rng = np.random.RandomState(1)

    clf = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=Bestset[0], min_samples_split=Bestset[1], min_samples_leaf=Bestset[2]),
        n_estimators=Bestadaboost[0], learning_rate=Bestadaboost[1], random_state=rng)
    # clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=Bestset[0], min_samples_split=Bestset[1], min_samples_leaf = Bestset[2]), n_estimators=Bestadaboost[0], learning_rate=Bestadaboost[1], random_state=rng)
    # clf = DecisionTreeRegressor(max_depth=Bestset[0], min_samples_split=Bestset[1], min_samples_leaf = Bestset[2])
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    #x_sample = range(len(y_pred))
    #plt.scatter(x_sample[:100], y_pred[:100], color='r')
    # print x_sample[:10]
    # print y_pred[:10]
    # print len(y_pred)
    #
    # with open('adaboost_result.csv', 'wb') as result_file:
    #     csv_writer = csv.writer(result_file)
    #     for i in range(len(y_pred)):
    #         csv_writer.writerow([y_pred[i]])
    res = calculate_mse(y_pred, y_test)
    dec_res.append(res)
    #
    # plt.scatter(x_sample[:100], y_test[:100], color='y')
    # plt.show()

print svm_res
print dec_res
x_sample = range(len(svm_res))
plt.scatter(x_sample, svm_res, color='b')
plt.scatter(x_sample, dec_res, color='r')
plt.ylabel('Mean Squared Error (MSE)')
plt.xlabel('Cluster ID')

text = iter(['SVM', 'Adaboost'])


mpl_fig = plt.gcf()
plotly_fig = tls.mpl_to_plotly( mpl_fig )

for dat in plotly_fig['data']:
    t = text.next()
    dat.update({'name': t, 'text':t})

plotly_fig['layout']['showlegend'] = True
py.plot(plotly_fig)
#plt.show()