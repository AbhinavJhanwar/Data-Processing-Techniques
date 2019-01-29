'''
Created on Aug 3, 2017

@author: abhinav.jhanwar
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
import csv


#CLASSIFICATION MODELS
url = "iris_data.csv"

with open(url) as csvFile:
    reader = csv.reader(csvFile)
    names = next(reader)
    
data = pd.read_csv(url)
feature_cols = names[0:-1] #names will be replaced by features directly taken from user selection
X = data[feature_cols]
y = data[names[-1]] #names replaced by target taken from user selection

#kf = KFold(n_splits=5, shuffle=False)

# print the contents of each training and testing set
# ^ - forces the field to be centered within the available space
# .format() - formats the string similar to %s or %n
# enumerate(sequence, start=0) - returns an enumerate object
#print('{:^61} {}'.format('Training set observations', 'Testing set observations'))

#for train_index, test_index in kf.split(range(1,26)):
#    print('{} {!s:^25}'.format(train_index, test_index))
    
# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
# k = 5 for KNeighborsClassifier   
k_scores = []
max_score=0
for k in range(1,31):
    knn = KNeighborsClassifier(n_neighbors=k)

    # Use cross_val_score function
    # We are passing the entirety of X and y, not X_train or y_train, it takes care of splitting the data
    # cv=10 for 10 folds
    # scoring='accuracy' for evaluation metric - although they are many
    # use StratifiedKFold for imbalanced target class
    kfold = StratifiedKFold(n_splits=10, random_state=7)
    scores = cross_val_score(knn, X, y, scoring='accuracy', cv=kfold).mean()
    #print(scores)
    k_scores.append(scores)
    # use average accuracy as an estimate of out-of-sample accuracy
    # numpy array has a method mean()
    if(k_scores[k-1]>=max_score):
        #print(k_scores[k-1])
        max_score = k_scores[k-1]
        #print(k)

#SIMILARLY GO FOR OTHER MODELS AND SELECT MODELS GIVING BEST SCORE        

#REGRESSION MODELS
url = "Advertising.csv"

with open(url) as csvFile:
    reader = csv.reader(csvFile)
    names = next(reader)
    
data = pd.read_csv(url)
feature_cols = names[0:-1] #names will be replaced by features directly taken from user selection
X = data[feature_cols]
y = data[names[-1]] #names replaced by target taken from user selection

model = LinearRegression()

# store scores in scores object
# we can't use accuracy as our evaluation metric since that's only relevant for classification problems
# RMSE is not directly available so we will use MSE
rmse_scores = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)).mean()

print(rmse_scores)

