'''
Created on Aug 1, 2017

@author: abhinav.jhanwar
'''
from sklearn.preprocessing import MinMaxScaler
import numpy as np


weights = np.array([[115], [140], [175]]).astype(float)
scaler = MinMaxScaler()
rescaled_weights = scaler.fit_transform(weights)
print(rescaled_weights)