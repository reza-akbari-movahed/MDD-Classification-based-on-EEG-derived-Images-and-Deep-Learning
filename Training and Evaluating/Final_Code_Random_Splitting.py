# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:06:41 2020

@author: Reza
"""

import numpy as np 
import tables as tb
from keras.models import Sequential
from keras.layers import Conv2D 
from keras.layers import LeakyReLU
from keras.layers import Softmax
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.layers import ReLU
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras.callbacks import EarlyStopping

def Evaluate(Predict,L_test):
    TP = 0 
    TN = 0 
    FP = 0 
    FN = 0
    for i in range(Predict.size):
        if (Predict[i]==1)and(L_test[i]==1):
            TP = TP + 1 
        elif (Predict[i]==0)and(L_test[i]==0):
            TN = TN + 1 
        elif (Predict[i]==1)and(L_test[i]==0):
            FP = FP + 1 
        else:
            FN = FN + 1 
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    Sensitivity = TP/(TP+FN)
    Specificity = TN/(TN+FP)
    FDR = FP/(TP+FP)
    F1 = (2*TP)/(2*TP+FP+FN)
    return Accuracy, Sensitivity, Specificity, FDR, F1 

def First_Model(Sh_inupt):
    model = Sequential()
    model.add(Conv2D(32,2,padding='same',input_shape=Sh_inupt))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32,3,padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32,5,padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Flatten())
    model.add(Dense(64))
    return model

def Second_Model(inputShape):
    inputs = Input(shape=inputShape)
    x = inputs
    x = Conv2D(8,(3,3), padding="same")(x)
    x = ReLU()(x)
    x = Conv2D(16,(3,3), padding="same")(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    model = Model(inputs,x)
    return model

def Predict_class(Predict):
    Predict_C = np.zeros(Predict.shape[0])
    a = Predict[:,0] > Predict[:,1]
    Predict_C[~a] = 1 
    return Predict_C
   
Batch_Size = 50 
Epochs = 10
num_classes = 2 

Mat_File = tb.open_file('Extracted_Images.mat')
Mat_File.root

Labels = Mat_File.root.Labels[:]
SP_Images = Mat_File.root.SP_Images[:]
FC_Images = Mat_File.root.FC_Images[:]
Labels = np.reshape(Labels,(Labels.shape[1]))
HC_Cases = Labels == 1
MDD_Cases = Labels == -1
Labels[HC_Cases] = 0
Labels[MDD_Cases] = 1
Num_HC_Cases = np.sum(Labels==0)
Num_MDD_Cases = np.sum(Labels==1)

if K.image_data_format() == 'channels_first':
    SP_Images = SP_Images.reshape(SP_Images.shape[0],SP_Images.shape[1],SP_Images.shape[2],SP_Images.shape[3])
    FC_Images = FC_Images.reshape(FC_Images.shape[0],1,FC_Images.shape[1],FC_Images.shape[2])
    input_SP_Images = (SP_Images.shape[1],SP_Images.shape[2],SP_Images.shape[3])
    input_FC_Images = (FC_Images.shape[1],FC_Images.shape[2],FC_Images.shape[3])
else:
    SP_Images = SP_Images.reshape(SP_Images.shape[0],SP_Images.shape[2],SP_Images.shape[3],SP_Images.shape[1])
    FC_Images = FC_Images.reshape(FC_Images.shape[0],FC_Images.shape[1],FC_Images.shape[2],1)
    input_SP_Images = (SP_Images.shape[1],SP_Images.shape[2],SP_Images.shape[3])
    input_FC_Images = (FC_Images.shape[1],FC_Images.shape[2],FC_Images.shape[3])
    
    
X_train_ST, X_test_ST, X_train_FC, X_test_FC, L_train_ST, L_test_ST = train_test_split(SP_Images,FC_Images,Labels,test_size=0.2)
L_train_cat = to_categorical(L_train_ST)
L_test_cat = to_categorical(L_test_ST)
        
model1 = First_Model(input_SP_Images)
model2 = Second_Model(input_FC_Images)

combinedInput = concatenate([model1.output, model2.output])
x = Dense(94)(combinedInput)
x = ReLU()(x)
x = Dense(num_classes)(x)
x = Softmax()(x)

model = Model(inputs=[model1.input, model2.input], outputs=x)

model.compile(loss=BinaryCrossentropy(),
              optimizer=Adam(learning_rate =  0.0003),
              metrics = ['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5)

model.fit(x=[X_train_ST, X_train_FC], y=L_train_cat, validation_split=0.2,
	epochs=Epochs, batch_size=Batch_Size, callbacks=[es])


Predict = model.predict([X_test_ST, X_test_FC],batch_size = Batch_Size)
Predict_test = Predict_class(Predict)
Accuracy, Sensitivity, Specificity, FDR, F1 = Evaluate(Predict_test,L_test_ST)  

