'''
Created on Apr 18, 2017

@author: abhinav.jhanwar
'''

#Import Library
import pandas as pd
import csv
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from time import time
import logging
import pylab as pl
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC

# SAMPLE 1
'''url = "iris_data.csv"

with open(url) as csvFile:
    reader = csv.reader(csvFile)
    names = next(reader)
    
data = pd.read_csv(url)
feature_cols = names[0:-1] #names will be replaced by features directly taken from user selection
X = data[feature_cols]
y = data[names[-1]] #names replaced by target taken from user selection
#print(X.shape)
#print(y.shape)

X_std = StandardScaler().fit_transform(X)

# Create PCA object 
pca = PCA(n_components=2)

# Fit and Apply dimensionality reduction on X
pc = pca.fit_transform(X_std)
print(pc) 

#The amount of variance that each PC explains
var= pca.explained_variance_ratio_
print(var)

validation_size = 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=42)
#Assumed you have training and test data set as train and test



#default value of k =min(n_sample, n_features)
# For Factor analysis
#fa= decomposition.FactorAnalysis()
# Reduced the dimension of training dataset using PCA
train_reduced = pca.fit_transform(train)
#Reduced the dimension of test dataset
test_reduced = pca.transform(test)'''

# SAMPLE 2
# Data of famous people's faces
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = faces.data
y = faces.target
target_names = faces.target_names
n_classes = target_names.shape[0]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = faces.image.shape

# For machine learning we use the data directly (as relative pixel
# position info is ignored by this model)
n_features = X.shape[1]

# the label to predict is the id of the person

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)





