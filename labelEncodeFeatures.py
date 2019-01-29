'''
Created on Jun 12, 2018

@author: abhinav.jhanwar
'''

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import os
from sklearn.externals import joblib

def labelEncodeData(data, features, target, path):
    # data: dataframe with the features to be encoded
    # categoricalFeatures: list of categorical features to be encoded
    # label encode categorical features and converting to dataframe with one hot encoding
    
    le = LabelEncoder()
    ohe = OneHotEncoder(sparse=False)
    for i, feature in enumerate(features):
        # create directory to save label encoding models
        if not os.path.exists(path+"{0}/TextEncoding".format(target)):
            os.makedirs(path+"{0}/TextEncoding".format(target))
            
        # perform label encoding
        le.fit(data[feature])
        # save the encoder
        joblib.dump(le, open(path+"{0}/TextEncoding/le_{1}.sav".format(target, feature), 'wb'))
        # transfrom training data
        data[feature] = le.transform(data[feature])
        
        # get classes & remove first column to elude from dummy variable trap
        columns = list(map(lambda x: feature+' '+str(x), list(le.classes_)))[1:]
        # save classes
        joblib.dump(columns, open(path+"{0}/TextEncoding/le_{1}_classes.sav".format(target, feature), 'wb'))
        # load classes
        columns = joblib.load(open(path+"{0}/TextEncoding/le_{1}_classes.sav".format(target, feature), 'rb'))
        
        # perform hot encoding
        ohe.fit(data[[feature]])
        # save the encoder
        joblib.dump(ohe, open(path+"{0}/TextEncoding/ohe_{1}.sav".format(target, feature),'wb'))
        # transfrom training data
        # removing first column of encoded data to elude from dummy variable trap
        tempData = ohe.transform(data[[feature]])[:, 1:]
        
        # create Dataframe with columns as classes
        tempData = pd.DataFrame(tempData, columns=columns)
        # create dataframe with all the label encoded categorical features along with hot encoding
        if i==0:
            encodedData = pd.DataFrame(data=tempData, columns=tempData.columns.values.tolist())
        else:
            encodedData = pd.concat([encodedData, tempData], axis=1)
        
    return encodedData

def labelEncodeTestData(encodings, data, features, target, path):
    # data: dataframe with the features to be encoded
    # categoricalFeatures: list of categorical features to be encoded
    # label encode categorical features and converting to dataframe with one hot encoding
    
    for i, feature in enumerate(features):
        # perform label encoding
        key = 'le_'+feature
        le = encodings[key]
        # transfrom test data
        data[feature] = le.transform(data[feature])
        
        # load classes
        columns = joblib.load(open(path+"{0}/TextEncoding/le_{1}_classes.sav".format(target, feature), 'rb'))
        
        # load oneHotEncoder
        key = 'ohe_'+feature
        ohe = encodings[key]
        # transfrom test data
        tempData = ohe.transform(data[[feature]])[:, 1:]
        
        # create Dataframe with columns as classes
        tempData = pd.DataFrame(tempData, columns=columns)
        # create dataframe with all the label encoded categorical features along with hot encoding
        if i==0:
            encodedData = pd.DataFrame(data=tempData, columns=tempData.columns.values.tolist())
        else:
            encodedData = pd.concat([encodedData, tempData], axis=1)
        
    return encodedData
    
    