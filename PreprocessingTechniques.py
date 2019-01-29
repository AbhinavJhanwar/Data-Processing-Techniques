'''
Created on Apr 17, 2017

@author: abhinav.jhanwar
'''

#CLASSIFICATION Problem

#Import Library
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from boruta import BorutaPy
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer, LabelEncoder, OneHotEncoder, Imputer, PolynomialFeatures
# use to remove outlier values
from scipy.stats import boxcox

def normalizeData(X):
    scaler = Normalizer().fit(X)
    normalizedX = scaler.transform(X)
    return normalizedX

def binarizeData(X):
    scaler = Binarizer(threshold=0.0).fit(X)
    binarizedX = scaler.transform(X)
    return binarizedX

def discretContinuous(target):
    uniquePercent = len(np.unique(target))/len(target)
    if(uniquePercent>0.2):
        return 'Regression'
    else:
        return 'Classification'
    
#hotEncode all categorical/binary/discrete data
def hotEncodeData(X, feature_cols): #feature_cols replaced by global variable features of feature or featureType
    encodedX = X
    enc = OneHotEncoder(sparse=False)
    for col in feature_cols:
        if discretContinuous(X[col])=='C':
            # creating an exhaustive list of all possible categorical values
            hotData=X[[col]]
            # Fitting One Hot Encoding on train data
            temp =  enc.fit_transform(hotData)
            # Changing the encoded features into a data frame with new column names
            temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in hotData[col]
                                            .value_counts().index])
            # In side by side concatenation index values should be same
            # Setting the index values similar to the X_train data frame
            temp=temp.set_index(hotData.index.values)
            # adding the new One Hot Encoded varibales to the train data frame
            encodedX = pd.concat([encodedX,temp],axis=1)
            # fitting One Hot Encoding on test data
    return encodedX

def getFeaturesRanking(X, y):
    rf= RandomForestRegressor(max_depth=5)
    #rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=5)
    try:
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=3, max_iter=20)
        feat_selector.fit(X.values, y.values)
        #print("names:",feat_selector.support_)
        #print("ranking:", feat_selector.ranking_)
        return feat_selector.ranking_
    except:
        return np.ones((len(names), ), dtype=np.int)

def computeBasicPreprocessing(data, names):
    global X, encodedClasses
    X = data
    le = LabelEncoder()
    encodeError = dict()
    labelEncodedFeatures = list()
    objectOutliers = list()
    encodedClasses = dict()
   
    for item in names:
        # Encoding only categorical variables
        if featureType[item]=='object':
            labelEncodedFeatures.append(item)
            X[item] = le.fit_transform(X[item])
            encodedClasses[item] = le.classes_.tolist()
            
            for EClass in range(0, len(le.classes_)):
                percent = X[item].value_counts()[EClass]*100/len(X[item])
                if percent<0.02:        #set percentage for outlier values
                    objectOutliers.append(item)
                    break
                
            if ' ?' in le.classes_:
                encodeError[item] = True
            else:
                encodeError[item] = False
              
    missingValueFeaturesContinuous, missingValueFeaturesDiscrete = getMissingValueFeatures(names, encodeError)
    outlierFeatures = getOutliersFeatures(X, names)+objectOutliers
    
    return labelEncodedFeatures, missingValueFeaturesContinuous, missingValueFeaturesDiscrete, outlierFeatures

def getMissingValueFeatures(names, encodeError):   #featureType is available as global 
    missingValueFeaturesContinuous, missingValueFeaturesDiscrete = list(),list()
    for item in names:      #replace with features
        if featureType[item]=='object' and encodeError[item]:
            missingValueFeaturesDiscrete.append(item)
        elif featureType[item]!='object':
            if discretContinuous(X[item])=='Regression':
                missingValueFeaturesContinuous.append(item)
            elif discretContinuous(X[item])=='Classification':
                missingValueFeaturesDiscrete.append(item)
    
    return missingValueFeaturesContinuous, missingValueFeaturesDiscrete

def getOutliersFeatures(X, names):  #X and featureType is global
    X = standardizeData(X)
    mean = X.mean()
    std = X.std()
    outlierFeaturesMean = list()        #to be replaced by mean
    outlierFeatures = list()
    
    for item in names:
        if featureType[item]!= 'object' and discretContinuous(X[item])=='Regression':
            outlierFeaturesMean.append(item)
        
    for item in outlierFeaturesMean:
        for val in X[item]:
            if (val<=mean[item]+3*std[item]) and (val>=mean[item]-3*std[item]):
                pass
            else:
                outlierFeatures.append(item)  
                break
    
    return outlierFeatures

def computeAdvancedPreprocessing(X, preprocessTechniques, missingValueFeaturesContinuous, missingValueFeaturesDiscrete, outlierFeatures):
    X = imputeMissingValues(X, missingValueFeaturesContinuous, missingValueFeaturesDiscrete)
    #X = X.dropna()  #remove NaN values if still present
    
    X = removeOutlierValues(X, outlierFeatures)
    
    if 'StandardizeFeatures' in preprocessTechniques:
        X = standardizeData(X)
        
    if 'ScaleFeatures' in preprocessTechniques:
        X = scaleData(X)
        
    if 'GeneratePolynomialFeatures' in preprocessTechniques:
        X = generatePolynomials(X)
        
    return X

def imputeMissingValues(X, missingValueFeaturesContinuous, missingValueFeaturesDiscrete):
    numContinuousImputed = pd.DataFrame()
    numDiscreteImputed = pd.DataFrame()
    objectImputed = pd.DataFrame()
    
    for item in missingValueFeaturesContinuous:     
        numContinuousImputed[item] = X[item]
        #print("continuous with error:",item)
        
    for item in missingValueFeaturesDiscrete:
        if featureType[item]=='object':
            objectImputed[item] = X[item]
            #print("object with error:",item)
        else:
            numDiscreteImputed[item] = X[item]
            #print("discrete with error",item)
        
    #strategy = ['mean', 'median', 'most_frequent']            
    imp = Imputer(missing_values=missingValue, strategy='mean', axis=0)
    numContinuousnp = imp.fit_transform(numContinuousImputed)
    numContinuousImputed = pd.DataFrame(data=numContinuousnp, columns=numContinuousImputed.columns.values.tolist())
        
    imp = Imputer(missing_values=missingValue, strategy='most_frequent', axis=0)
    numDiscretenp = imp.fit_transform(numDiscreteImputed)
    numDiscreteImputed = pd.DataFrame(data=numDiscretenp, columns=numDiscreteImputed.columns.values.tolist())
    
    imp = Imputer(missing_values=0, strategy='most_frequent', axis=0)
    objectImputednp = imp.fit_transform(objectImputed)
    objectImputed = pd.DataFrame(data=objectImputednp, columns=objectImputed.columns.values.tolist())
    
    for item in numContinuousImputed.columns.values.tolist():
        X[item] = numContinuousImputed[item]
    
    for item in numContinuousImputed.columns.values.tolist():
        X[item] = numContinuousImputed[item]
        
    for item in numContinuousImputed.columns.values.tolist():
        X[item] = numContinuousImputed[item]
    
    return X

def removeOutlierValues(X, outlierFeatures):
    mean_X = X.mean()
    X_standard = standardizeData(X)
    mean = X_standard.mean()
    std = X_standard.std()
    
    for item in outlierFeatures:
        if featureType[item]=='object':
            for EClass in range(0, len(encodedClasses[item])):
                percent = X[item].value_counts()[EClass]*100/len(X[item])
                if percent<0.02:        #set percentage for outlier values
                    X[item] = X[item].replace(EClass, X[item].mode()[0])
        else:
            i=0
            for val in X_standard[item]:
                if (val<=mean[item]+3*std[item]) and (val>=mean[item]-3*std[item]):
                    pass
                else:
                    X[item][i] = mean_X[item]
                i+=1
    return X

def generatePolynomials(X):
    poly = PolynomialFeatures(degree=2)
    polyX = poly.fit_transform(X)
    return polyX
     
def scaleData(X):
    scaler = MinMaxScaler(feature_range=(0,1))
    rescaledX = scaler.fit_transform(X)
    return rescaledX

def standardizeData(X):
    scaler = StandardScaler().fit(X)
    rescaledX = scaler.transform(X)
    rescaledX = pd.DataFrame(data=rescaledX,columns=X.columns.values.tolist())
    return rescaledX

url = "adult.csv"
data = pd.read_csv(url)
names = data.columns.values.tolist()
featureType = data.dtypes
missingValue = np.NaN

#imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
#X = imp.fit_transform(X)
#X = pd.DataFrame(data=objectDF, columns=objectDFFeatures)
#imputeMissingValues(X, featureType, encodeError)
#X = hotEncodeData(X, feature_cols)
#X = generatePolynomials(X)
#X = scaleData(X)
#ranking = getFeaturesRanking(X, y)
outlierFeatures = getOutliersFeatures(X)
X = removeOutlierValues(X, outlierFeatures)
print(X.tail(50))

'''
validation_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=0)

model = LinearRegression()#KNeighborsClassifier(n_neighbors=5)#LogisticRegression()#
model.fit(X_train, y_train)
#0.815292491939
#accuracy = metrics.accuracy_score(y_test, model.predict(X_test))
accuracy = np.sqrt((-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)).mean())
print(accuracy)

'''