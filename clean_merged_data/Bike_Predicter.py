import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import AdaBoostRegressor
from pylab import *
from sklearn import cross_validation
from sklearn.tree import DecisionTreeRegressor
from sklearn import grid_search
import time
from datetime import datetime
# Define a bicycle predicter

class Bikepredictor:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    # def TrainDateTransform(self, date):
     #    for i in range(date.shape[0]):
     #        year_day = date[i].astype(datetime).timetuple().tm_yday
     #        self.data[i][0] = year_day
	# # date(yday)	year	month	hour	holiday	weekday	working	weather_type	temp	feels_like	humidity	windspeed	count
	# # 0				1		2		3		4		5		6		7				8		9			10			11			12
	# # totalday : 	if year == 0:
	# #					totalday = date
	# #				else
	# #					totalday = date + 365
    # def TestDateTransform(self, testdate, testdata):
     #    for i in range(testdate.shape[0]):
     #        year_day = testdate[i].astype(datetime).timetuple().tm_yday
     #        testdata[i][0] = year_day
     #    return testdata
	# # date(yday)	year	month	hour	holiday	weekday	working	weather_type	temp	feels_like	humidity	windspeed
	# # 0				1		2		3		4		5		6		7				8		9			10			11
	# # totalday : 	if year == 0:
	# #					totalday = date
	# #				else
	# #					totalday = date + 365
#     def TrainDataNormalize(self):
# 	# Normalize temp & feels_like & windspeed
#         self.data[:,8] = self.data[:,8]/(max(self.data[:,8]))
#         self.data[:,9] = self.data[:,9]/(max(self.data[:,9]))
#         self.data[:,11] = self.data[:,11]/(max(self.data[:,11]))
#     # Calculate rmsd
#     def rmsd(self, estimatevalue, realvalue):
#         error = estimatevalue - realvalue
#         rm = sum(error**2)
#         result = (rm/estimatevalue.shape[0])**0.5
#         return result
#     # Calculate rmsle, assert that the shape of estimatevalue & realvalue are same
#     def rmsle(self, estimatevalue, realvalue):
# #        logsum = 0.0
# #        for i in range(estimatevalue.shape[0]):
# #            temp = math.log(estimatevalue[i] + 1) - math.log(realvalue[i] + 1)
# #            temp = temp**2
# #            logsum = logsum + temp
#         estimatevalue = np.array(estimatevalue)
#         realvalue = np.array(realvalue)
#         errorvalue = np.log(estimatevalue + 1)-np.log(realvalue + 1)
#         errorvalue = errorvalue**2
#         result = mean(errorvalue)
#         return sqrt(result)
#        return logsum/estimatevalue.shape[0]
#     # testing the Training data and tuning the hyperparameter
#     def crossvalidation(self):
#         fold_num = 10
#         error = 0
#         kf = cross_validation.KFold(self.data.shape[0], n_folds=fold_num)
#         maxTreeDepth = range(11,20,2)
#         minTreeSplit = range(1,10,2)
#         minTreeNodeSample = range(2,10,3)
#         Bestset = [0,0,0]
#         Errorset = []
#         print "Start CV"
#         for depth in maxTreeDepth:
#             for split in minTreeSplit:
#                 for nodenum in minTreeNodeSample:
#                     avgerror = 0
#                     avgrmsderror = 0
#                     for train_index, test_index in kf:
#                         # Get Train set & Test set
#                         x_train, x_test = self.data[train_index,:12], self.data[test_index,:12]
#                         y_train, y_test = self.data[train_index,12], self.data[test_index,12]
#                         # Acquire the Prediction of the data
#                         clf = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=nodenum, min_samples_split=split)
#                         # clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=depth, min_samples_leaf=nodenum, min_samples_split=split), n_estimators=300, random_state=rng)
#                         clf = clf.fit(x_train, y_train)
#                         estimatevalue = clf.predict(x_test)
#                         #rmsderror = self.rmsd(estimatevalue, y_test)
#                         error = self.rmsle(estimatevalue, y_test)
#                         #print error
#                         avgerror = avgerror + error
#                         #avgrmsderror = avgrmsderror + rmsderror
#                     avgerror = avgerror/fold_num
#                     #avgrmsderror = avgrmsderror/fold_num
#                     #print "/t", avgerror
#                     #print "Depth is",depth,"Split is", split,"Node Number is",nodenum, "Average error is", avgerror, "Min error is", min(Errorset)
#                     Errorset.append(avgerror)
#                     print "Depth is",depth,"Split is", split,"Node Number is",nodenum, "Average error is", avgerror, "Min error is", min(Errorset)
#                     if min(Errorset)==avgerror:
#                         Bestset[0] = depth
#                         Bestset[1] = split
#                         Bestset[2] = nodenum
#                         #print avgrmsderror
#                     #print Bestset
#
#         print min(Errorset)
#         print "CV Finished"
#         return Bestset
#     # Turing the hyperparameter in Adaboost Regression
#     def adaboosttuner(self, Bestset):
#         fold_num = 10
#         error = 0
#         kf = cross_validation.KFold(self.data.shape[0], n_folds=fold_num)
#         #learnRateSet = np.linspace(0.1,1,20)
#         learnRateSet = [0.05]
#         Treenumset = range(200,500,30)
#         rng = np.random.RandomState(1)
#         Errorset = []
#         Bestadaboost = [0,0]
#         print "Start Tune Adaboost"
#         for treenum in Treenumset:
#             for learnrate in learnRateSet:
#                 avgerror = 0
#                 for train_index, test_index in kf:
#                     # Get Train set & Test set
#                     x_train, x_test = self.data[train_index,:12], self.data[test_index,:12]
#                     y_train, y_test = self.data[train_index,12], self.data[test_index,12]
#                     clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=Bestset[0], min_samples_leaf=Bestset[1], min_samples_split=Bestset[2]), n_estimators=treenum, learning_rate=learnrate, random_state=rng)
#                     clf = clf.fit(x_train, y_train)
#                     estimatevalue = clf.predict(x_test)
#                     error = self.rmsle(estimatevalue, y_test)
#                     avgerror = avgerror + error
#                 avgerror = avgerror/fold_num
#                 print avgerror
#                 Errorset.append(avgerror)
#                 if min(Errorset)==avgerror:
#                     Bestadaboost[0] = treenum
#                     Bestadaboost[1] = learnrate
#                 print Bestadaboost
#         print min(Errorset)
#         print "Adaboost tune Finish!"
#         return Bestadaboost
    
    # Decision tree
    def BicyclePredict(self, X_test):
        print "Prediction Start!"
        Bestset = [19,3,2]
        Bestadaboost = [75, 0.05]
        rng = np.random.RandomState(1)

        clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=Bestset[0], min_samples_split=Bestset[1], min_samples_leaf = Bestset[2]), n_estimators=Bestadaboost[0], learning_rate=Bestadaboost[1], random_state=rng)
        #clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=Bestset[0], min_samples_split=Bestset[1], min_samples_leaf = Bestset[2]), n_estimators=Bestadaboost[0], learning_rate=Bestadaboost[1], random_state=rng)
        # clf = DecisionTreeRegressor(max_depth=Bestset[0], min_samples_split=Bestset[1], min_samples_leaf = Bestset[2])
        clf = clf.fit(self.X, self.y)

        estimatevalue = clf.predict(X_test)
        print "Prediction Finished!"
        return estimatevalue




