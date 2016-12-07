import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import csv
import time
import math
import calendar
from datetime import datetime


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


df = pd.read_csv('data_by_cluster/cluster_0_data_with_weather.csv')
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
svc = svm.SVC(C=1, kernel='linear')
clf = svc.fit(X_train, y_train)
X_test = df_test_feature.drop('count', 1).values
# print X_test[:10]
y_test = df_test_feature['count'].values
# print y_test[:10]

y_pred = clf.predict(X_test)
x_sample = range(len(y_pred))
plt.scatter(x_sample[:100], y_pred[:100], color='b')
print x_sample[:10]
print y_pred[:10]
print len(y_pred)

with open('svm_result.csv', 'wb') as result_file:
    csv_writer = csv.writer(result_file)
    for i in range(len(y_pred)):
        csv_writer.writerow([y_pred[i]])
print calculate_mse(y_pred, y_test)

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
x_sample = range(len(y_pred))
plt.scatter(x_sample[:100], y_pred[:100], color='r')
print x_sample[:10]
print y_pred[:10]
print len(y_pred)

with open('adaboost_result.csv', 'wb') as result_file:
    csv_writer = csv.writer(result_file)
    for i in range(len(y_pred)):
        csv_writer.writerow([y_pred[i]])
print calculate_mse(y_pred, y_test)

plt.scatter(x_sample[:100], y_test[:100], color='y')
plt.show()
# tst_data_date = tst_data['date'].values
# tst_data_other = tst_data.values
# print tst_data_date
# print tst_data_other
# BP = Bikepredictor(trn_data) #save data in the predictor
# print BP.data[:10]
# BP.TrainDateTransform(trn_data_date) #convert date to day in a year
# print BP.data[:10]
# tst_data = BP.TestDateTransform(tst_data_date,tst_data_other) #convert date to day in a year
# print tst_data[:10]
# tst_result = BP.BicyclePredict(tst_data)
# with open('midterm_result.csv', 'wb') as result_file:
#     csv_writer = csv.writer(result_file)
#     for i in range(len(tst_result)):
#         csv_writer.writerow([tst_result[i]])
# df_train = extract_2011(df)
# df_train_feature = create_features(df_train)
# df_test = extract_2012(df)
# X_train = df_train_feature.drop('count', 1)
# y_train = df_train_feature['count']

# df_test_feature = create_features(df_test)
# X_test = df_test_feature.drop('count', 1)
# y_test = df_test_feature['count']
# y_pred = clf.predict(X_test)
# print y_pred
# print clf.score(X_test, y_test)


# df_feature = normalize(df_feature)
# X = df_feature.drop('count', 1)
# y = df_feature['count']
# from sklearn.model_selection import train_test_split
# print X.head()
# print y.head()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


# from sklearn.model_selection import KFold, cross_val_score
# k_fold = KFold(n_splits=4)
#
#
# print clf.score(X_test, y_test)
#
# from sklearn.model_selection import ShuffleSplit
# cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
# scores = cross_val_score(svc, X, y, cv=5)
# print scores
# score2 = cross_val_score(svc, X, y, cv=cv)
# print score2
