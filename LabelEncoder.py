'''
Created on Apr 17, 2017

@author: abhinav.jhanwar
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import csv

url = "iris_data.csv"

with open(url) as csvFile:
    reader = csv.reader(csvFile)
    names = next(reader)
    
data = pd.read_csv(url)
feature_cols = names[0:-1] #names will be replaced by features directly taken from user selection
X = data[feature_cols]
y = data[names[-1]] #names replaced by target taken from user selection

# limit to categorical data using df.select_dtypes()
data = data.select_dtypes(include=[object])
print(data.columns)

# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()

# 2/3. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
X_2 = data.apply(le.fit_transform)
print(X_2.head())

