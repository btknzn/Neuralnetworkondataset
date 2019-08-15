#Artificall Bussines Problem to solve problem /
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
#taking important feature from dataset
y = dataset.iloc[:,13].values
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:,1]  = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_x_2 = LabelEncoder()
X[:,2]  = labelencoder_x_2.fit_transform(X[:,2])
#this part we made encoding and for example france = 1 0 0 but after
#X = X[:,1:] part, france will be for us 0 0 
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#spliting the dataset into Training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Importing the Keras Libraries and packages
import keras 
from keras.models import Sequential
from keras.layers import Dense
# initialising the ANN

classsifier = Sequential()
#adding Laters 
classsifier.add(Dense(output_dim=6,init ='uniform',activation ='relu', input_dim=11))
classsifier.add(Dense(output_dim=6,init ='uniform',activation ='relu'))
classsifier.add(Dense(output_dim=1,init ='uniform',activation ='sigmoid'))

#compling the ANN
classsifier.compile(optimizer = 'adam', loss='binary_crossentropy',metrics =['accuracy'] )

classsifier.fit(X_train, y_train,batch_size=10,nb_epoch=100)

y_pred = classsifier.predict(X_test)
yz_pred = (y_pred > 0.5)
z