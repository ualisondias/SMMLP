#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 23:11:23 2021

@author: ualisondias
"""

import  os
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, models
from    tensorflow.keras import regularizers
from    tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from    tensorflow.keras.models import Sequential
from    tensorflow.keras.layers import Dense, Activation
from    tensorflow.keras.utils import to_categorical
#from  tensorflow.keras.layer.core import Activation
import  numpy as np
import  pandas as pd
from    saphyra import PatternGenerator
from    Gaugi import load
import  numpy as np
from    sklearn.model_selection import StratifiedKFold, KFold
from    sklearn import preprocessing
from    scipy.io import savemat
import    scipy.io
    
import  numpy as np
import  math
from numpy.linalg import inv

import  matplotlib.pyplot as plt

import time as timee

    
import  tensorflow.keras
from    tensorflow.keras import backend as K
from    tensorflow.keras.datasets import cifar10
# from tensorflow.keras.utils import np_utils
from    tensorflow.keras.models import Sequential
from    tensorflow.keras.layers import Dense, Dropout, Flatten
from    tensorflow.keras.layers import Conv2D, MaxPooling2D
from    tensorflow.keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from    tensorflow.keras.models import model_from_json
from    tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay



# file = '/afs/cern.ch/user/u/uferreir/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et2_eta4.npz'
mat5 = scipy.io.loadmat('/Users/ualisondias/Documents/CERN/SMMLP/dados_completos.mat')
# mat5 = scipy.io.loadmat('/home/dias/Documents/CERN/SMMLP/Results_appendiciti/dados_completos.mat')
# mat5 = scipy.io.loadmat('/home/ualisondias/Documents/SMMLP/Results_appendiciti/dados_completos.mat')

def update_learning_rate(epoch, learning_rate, n_iterations, y_pred, typee = 'mlp'): #####33
    global update  
    global n_update
    global Gamma
    global times
    global mi
        
    # print(model3.layers[0].get_weights()[1])
    
    if epoch==0:
            mi = learning_rate
    if typee == 'mse':
        globals()[f"erro_{model}"][times,epoch] = mean_squared_error(y_train,y_pred)
        # globals()[f"erro_{model}"][times,epoch] = (np.mean(abs(y_train-y_pred))**2)
        globals()[f"accuracyy_{model}"][times,epoch] = accuracy_score(np.argmax(y_train, axis=1), np.argmax(y_pred, axis=1))  
    else:
        globals()[f"erro_{model}"][times,epoch] = mean_absolute_error(y_train,y_pred)
        # globals()[f"erro_{model}"][times,epoch] = (np.mean(abs(y_train-y_pred)))
        globals()[f"accuracyy_{model}"][times,epoch] = accuracy_score(np.argmax(y_train, axis=1), np.argmax(y_pred, axis=1))        
        

    if typee == 'mlp':
        mi = learning_rate
    
    if typee == 'mse':
        # locals()[f"erro_{model}"] = loss[epoch -1]
        # if epoch==0:
        #     mi = learning_rate
        # if locals()[f"erro_{model}"] > Gamma:
        if globals()[f"erro_{model}"][times,epoch] > Gamma[times]:    
            #mi = 1 - (Gamma/erro*10)
            # mi = (Gamma/erro*0.01)
            mi = (Gamma[times]/(globals()[f"erro_{model}"][times,epoch])*(10**(-2)))
            globals()[f"update_{model}"][times] = globals()[f"update_{model}"][times] + 1
        else:
            mi = 0  
            globals()[f"n_update_{model}"][times] = globals()[f"n_update_{model}"][times] + 1

    if typee == 'sm': # and epoch >= 10:
        # erro = loss[epoch -1]
        # if epoch==0:
        #     mi = learning_rate
        # if erro > Gamma:
        if globals()[f"erro_{model}"][times,epoch] > Gamma[times]:    
            #mi = 1 - (Gamma/erro*10)
            # mi = (Gamma/erro*0.01)
            # mi = (Gamma[times]/globals()[f"erro_{model}"][times,epoch])
            mi = (Gamma[times]/(globals()[f"erro_{model}"][times,epoch])*(10**(-1)))
            globals()[f"update_{model}"][times] = globals()[f"update_{model}"][times] + 1
        else:
            mi = 0  
            globals()[f"n_update_{model}"][times] = globals()[f"n_update_{model}"][times] + 1

    if typee =='vs':
      # erro = loss[epoch -1]
      # mi = learning_rate
      # if epoch==0:
      #     mi = learning_rate
      # # if erro > Gamma:
      if globals()[f"erro_{model}"][times,epoch] > Gamma[times]:    
          mi = mi * (learning_rate*(10**(3)))
          globals()[f"update_{model}"][times] = globals()[f"update_{model}"][times] + 1
      else:
          mi = mi / (learning_rate*(10**(3)))
          globals()[f"n_update_{model}"][times] = globals()[f"n_update_{model}"][times] + 1     
              
    if typee =='mvs':
      # erro[times,epoch] = loss[epoch -1]
      # mi = learning_rate
      # if epoch==0:
      #     mi = learning_rate
      # # if erro[times,epoch][times,epoch] > Gamma:
      if globals()[f"erro_{model}"][times,epoch] > Gamma[times]:    
          mi = mi * (learning_rate*(10**(3))) 
          globals()[f"update_{model}"][times] = globals()[f"update_{model}"][times] + 1
      else:
          mi = 0  
          globals()[f"n_update_{model}"][times] = globals()[f"n_update_{model}"][times] + 1  
    
    if typee =='vsa':
      # erro[times,epoch][times,epoch] = loss[epoch -1]
      # mi = learning_rate
      # if epoch==0:
      #     mi = learning_rate
      # # if erro[times,epoch][times,epoch] > Gamma:
      if globals()[f"erro_{model}"][times,epoch] > Gamma[times]:    
          mi = mi + (learning_rate*(10**(-1)))
          globals()[f"update_{model}"][times] = globals()[f"update_{model}"][times] + 1
      else:
          mi = mi - (learning_rate*(10**(-1))) 
          globals()[f"n_update_{model}"][times] = globals()[f"n_update_{model}"][times] + 1  
    
    if typee =='mvsa':
      # erro[times,epoch] = loss[epoch -1]
      # mi = learning_rate
      # if epoch==0:
      #     mi = learning_rate
      # # if erro[times,epoch] > Gamma:
      if globals()[f"erro_{model}"][times,epoch] > Gamma[times]:
          mi = mi + (learning_rate*(10**(-1)))
          globals()[f"update_{model}"][times] = globals()[f"update_{model}"][times] + 1
      else:
          mi = 0
          globals()[f"n_update_{model}"][times] = globals()[f"n_update_{model}"][times] + 1

    if typee == 'smnn': # and epoch >= 10:
        # erro = loss[epoch -1]
        # if epoch==0:
        #     mi = learning_rate
        # if erro > Gamma:
        if globals()[f"erro_{model}"][times,epoch] > Gamma[times]:    
            #mi = 1 - (Gamma/erro*10)
            # mi = (Gamma/erro*0.01)
            # mi = (Gamma[times]/globals()[f"erro_{model}"][times,epoch])
            mii = (Gamma[times]/(globals()[f"erro_{model}"][times,epoch])*(10**(-1)))
            dist = np.linalg.norm(learning_rate-mii)
            mi = mi + dist*(0.00001)
            globals()[f"update_{model}"][times] = globals()[f"update_{model}"][times] + 1
        else:
            mi = 0
            globals()[f"n_update_{model}"][times] = globals()[f"n_update_{model}"][times] + 1    
          
    if typee == 'emse':
        # locals()[f"erro_{model}"] = loss[epoch -1]
        # if epoch==0:
        #     mi = learning_rate
        # if locals()[f"erro_{model}"] > Gamma:
        if globals()[f"erro_{model}"][times,epoch] > Gamma[times]:    
            #mi = 1 - (Gamma/erro*10)
            # mi = (Gamma/erro*0.01)
            mi = (Gamma[times]/(globals()[f"erro_{model}"][times,epoch])*(10**(-2)))
            globals()[f"update_{model}"][times] = globals()[f"update_{model}"][times] + 1
        else:
            mi = 0  
            globals()[f"n_update_{model}"][times] = globals()[f"n_update_{model}"][times] + 1
                      
        if mi < IL:
            mi = IL
        if mi > SL:
            mi = SL           

    if typee =='emvs':
      # erro[times,epoch] = loss[epoch -1]
      # mi = learning_rate
      # if epoch==0:
      #     mi = learning_rate
      # # if erro[times,epoch][times,epoch] > Gamma:
        if globals()[f"erro_{model}"][times,epoch] > Gamma[times]:    
            mi = mi * (learning_rate*(10**(3))) 
            globals()[f"update_{model}"][times] = globals()[f"update_{model}"][times] + 1
        else:
            mi = 0  
            globals()[f"n_update_{model}"][times] = globals()[f"n_update_{model}"][times] + 1  
    
        if mi < IL:
            mi = IL
        if mi > SL:
            mi = SL

    if typee =='emvsa':
      # erro[times,epoch] = loss[epoch -1]
      # mi = learning_rate
      # if epoch==0:
      #     mi = learning_rate
      # # if erro[times,epoch] > Gamma:
        if globals()[f"erro_{model}"][times,epoch] > Gamma[times]:
            mi = mi + (learning_rate*(10**(-1)))
            globals()[f"update_{model}"][times] = globals()[f"update_{model}"][times] + 1
        else:
            mi = 0
            globals()[f"n_update_{model}"][times] = globals()[f"n_update_{model}"][times] + 1          
        
        if mi < IL:
            mi = IL
        if mi > SL:
            mi = SL

    if typee =='evs':
      # erro = loss[epoch -1]
      # mi = learning_rate
      # if epoch==0:
      #     mi = learning_rate
      # # if erro > Gamma:
      if globals()[f"erro_{model}"][times,epoch] > Gamma[times]:    
          mi = mi * (learning_rate*(10**(3)))
          globals()[f"update_{model}"][times] = globals()[f"update_{model}"][times] + 1
      else:
          mi = mi / (learning_rate*(10**(3)))
          globals()[f"n_update_{model}"][times] = globals()[f"n_update_{model}"][times] + 1     
          
      if mi < IL:
          mi = IL
      if mi > SL:
          mi = SL
          
    if typee =='evsa':
      # erro[times,epoch][times,epoch] = loss[epoch -1]
      # mi = learning_rate
      # if epoch==0:
      #     mi = learning_rate
      # # if erro[times,epoch][times,epoch] > Gamma:
        if globals()[f"erro_{model}"][times,epoch] > Gamma[times]:    
            mi = mi + (learning_rate*(10**(-1)))
            globals()[f"update_{model}"][times] = globals()[f"update_{model}"][times] + 1
        else:
            mi = mi - (learning_rate*(10**(-1))) 
            globals()[f"n_update_{model}"][times] = globals()[f"n_update_{model}"][times] + 1  
    
        if mi < IL:
            mi = IL
        if mi > SL:
            mi = SL

    if typee == 'esm': # and epoch >= 10:
        # erro = loss[epoch -1]
        # if epoch==0:
        #     mi = learning_rate
        # if erro > Gamma:
        if globals()[f"erro_{model}"][times,epoch] > Gamma[times]:    
            #mi = 1 - (Gamma/erro*10)
            # mi = (Gamma/erro*0.01)
            # mi = (Gamma[times]/globals()[f"erro_{model}"][times,epoch])
            mi = (Gamma[times]/(globals()[f"erro_{model}"][times,epoch])*(10**(-1)))
            globals()[f"update_{model}"][times] = globals()[f"update_{model}"][times] + 1
        else:
            mi = 0  
            globals()[f"n_update_{model}"][times] = globals()[f"n_update_{model}"][times] + 1
    
        if mi < IL:
            mi = IL
        if mi > SL:
            mi = SL

    if typee == 'esmnn': # and epoch >= 10:
        # erro = loss[epoch -1]
        # if epoch==0:
        #     mi = learning_rate
        # if erro > Gamma:
        if globals()[f"erro_{model}"][times,epoch] > Gamma[times]:    
            #mi = 1 - (Gamma/erro*10)
            # mi = (Gamma/erro*0.01)
            # mi = (Gamma[times]/globals()[f"erro_{model}"][times,epoch])
            mii = (Gamma[times]/(globals()[f"erro_{model}"][times,epoch])*(10**(-1)))
            dist = np.linalg.norm(learning_rate-mii)
            mi = mi + dist*(0.00001)
            globals()[f"update_{model}"][times] = globals()[f"update_{model}"][times] + 1
        else:
            mi = 0
            globals()[f"n_update_{model}"][times] = globals()[f"n_update_{model}"][times] + 1
    
        if mi < IL:
            mi = IL
        if mi > SL:
            mi = SL            

    return mi

def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


########################################################################

def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


########################################################################

def accuracy_score(y_true, y_pred):
   
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

########################################################################


# Define a new class and instantiate
class NewClass(object): pass


########################################################################

class CrossEntropy():
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)

#######################################################################


class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))
    
#########################################################################

class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)


########################################################################

class MultilayerPerceptron():
    
    
    def __init__(self, n_hidden, n_iterations=3000, learning_rate=0.1, model='normal'):
        self.model = model
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = Sigmoid()
        self.output_activation = Softmax()
        self.loss = CrossEntropy()        

    def _initialize_weights(self, X, y):
        
        n_samples, n_features = X.shape
        _, n_outputs = y.shape
        
        # Hidden layer
        limit   = 1 / math.sqrt(n_features)
        self.W  = np.random.uniform(-limit, limit, (n_features, self.n_hidden))
        self.w0 = np.zeros((1, self.n_hidden))
        
        # Output layer
        limit   = 1 / math.sqrt(self.n_hidden)
        self.V  = np.random.uniform(-limit, limit, (self.n_hidden, n_outputs))
        self.v0 = np.zeros((1, n_outputs))
        
    
    def fit(self, X, y):

        self._initialize_weights(X, y)      
        for i in range(self.n_iterations):
            
            # ..............
            #  Forward Pass
            # ..............

            # HIDDEN LAYER
            hidden_input = X.dot(self.W) + self.w0
            hidden_output = self.hidden_activation(hidden_input)
            # OUTPUT LAYER
            output_layer_input = hidden_output.dot(self.V) + self.v0
            y_pred = self.output_activation(output_layer_input)
            
            ############################################################

            # ...............
            #  Backward Pass
            # ...............

            # OUTPUT LAYER
            # Grad. with respect to input of output layer
            grad_wrt_out_l_input = self.loss.gradient(y, y_pred) * self.output_activation.gradient(output_layer_input)
            grad_v = hidden_output.T.dot(grad_wrt_out_l_input)
            grad_v0 = np.sum(grad_wrt_out_l_input, axis=0, keepdims=True)
            
            # HIDDEN LAYER
            # Grad. with respect to input of hidden layer
            grad_wrt_hidden_l_input = grad_wrt_out_l_input.dot(self.V.T) * self.hidden_activation.gradient(hidden_input)
            grad_w = X.T.dot(grad_wrt_hidden_l_input)
            grad_w0 = np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)
            
            ############################################################

            # ## SetMembership
            # if self.model=='normal':

            #     Gamma = 0.0005    # upper bound for the error modulus
            #     erro = (np.mean(abs(y-y_pred)))
            #     if i==0:
            #         mi = self.learning_rate
            #     if abs(erro) > Gamma:
            #         mi = (Gamma/abs(erro)) 
            #     else:
            #         mi = 0  

            # elif self.model=='mse':
            
            #     Gamma = 0.0003    # upper bound for the error modulus
            #     erro[i] = (np.mean(abs(y-y_pred))**2)
            #     if i==0:
            #         mi = self.learning_rate
            #     if abs(erro[i]) > Gamma:
            #         mi = (Gamma/abs(erro[i])) 
            #     else:
            #         mi = 0  


            # Update weights (by gradient descent)
            # Move against the gradient to minimize loss
            mi = update_learning_rate(i, self.learning_rate, self.n_iterations, y_pred, typee = self.model)
            self.V  -= mi * grad_v
            self.v0 -= mi * grad_v0
            self.W  -= mi * grad_w
            self.w0 -= mi * grad_w0
            globals()[f"lr_s_{model}"][times,i] = mi
            
            
 

    
    def predict(self, X):
        hidden_input = X.dot(self.W) + self.w0
        hidden_output = self.hidden_activation(hidden_input)
        output_layer_input = hidden_output.dot(self.V) + self.v0
        y_pred = self.output_activation(output_layer_input)
        return y_pred
    



##########################################################

def Model_MLP(): 
    
    model = Sequential()
    # model.add(Dense(7))
    model.add(Dense(7, activation='tanh'))
    model.add(Dense(5, activation='tanh'))
    model.add(Dense(2, activation='sigmoid'))
    # print(model.summary())
    # model.summary()
    return(model)



# define step function
class LossHistory(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        # self.weights = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        # print(model3.layers[0].get_weights()[0])
        print('lr:', self.lr[len(self.losses) -1])
        


def step_decay(epoch):
    loss = loss_history.losses
    if epoch == 0:
        LEARNING_RATE = 0.0002
        loss = [0]
    elif epoch == 1:
            LEARNING_RATE = 0.0002
    else:
        LEARNING_RATE = loss_history.lr[epoch - 2]
        LEARNING_RATE= update_learning_rate(epoch, loss, LEARNING_RATE, typee = 'normal')
        LEARNING_RATE = np.float32(LEARNING_RATE) 
    return LEARNING_RATE


def norm1( data ):
    norms = np.abs( data.sum(axis=1) )
    norms[norms==0] = 1
    return data/norms[:,None]

def my_loss_fn(y_true, y_pred):
    squared_difference = np.abs(y_true - y_pred)
    return squared_difference  # Note the `axis=-1`


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                "Naive Bayes", "QDA"]
                                  

# names = ["Linear SVM"]


# models = ["mse", "mlp", "mvs", "mvsa", "vs", "vsa", "sm"]
models = ["mse", "mlp", "mvs", "mvsa", "vs", "vsa", "sm", "smnn", "emse", "emvs", "emvsa", "evs", "evsa", "esm", "esmnn"]
# models = ["mvs"]


matrixA = {}
# SMMLP = {}

# matrix = NewClass()
# matrix.SMMLP = SMMLP

# matrixA = NewClass()
# matrixA = {'Scikit': { }}




        
# # Gamma = 0.0005    # upper bound for the error modulus
# Gamma = np.linspace(0.5, 0.5, num=3)

epochs = 1000
# time = 3
time = 50
# time = Gamma.size    
    

Accuracy = np.zeros((time * len(models) , len(names)))
stime_scikit = np.zeros((time * len(models) , len(names)))
mean_Accuracy = np.zeros((time * len(models) , len(names)))
kappa = np.zeros((time * len(models) , len(names)))
fscore = np.zeros((time * len(models) , len(names)))
mse = np.zeros((time * len(models) , len(names)))
Accuracy_train = np.zeros((time * len(models) , len(names)))
mean_Accuracy_train = np.zeros((time * len(models) , len(names)))
kappa_train = np.zeros((time * len(models) , len(names)))
fscore_train = np.zeros((time * len(models) , len(names)))
mse_train = np.zeros((time * len(models) , len(names)))
# results2 = np.zeros((len(names),time))
# mean_Accuracy = np.zeros((time,time * len(models) ))
# mean_Accuracy_train = np.zeros((time,time * len(models) ))
std_Accuracy = np.zeros((time * len(models) , len(names)))
std_Accuracy_train = np.zeros((time * len(models) , len(names)))
mean_mse = np.zeros((time * len(models) , len(names)))
mean_mse_train = np.zeros((time * len(models) , len(names)))
std_mse = np.zeros((time * len(models) , len(names)))
std_mse_train = np.zeros((time * len(models) , len(names)))

for model in models:
    locals()[f"accuracy_test_{model}"] = np.zeros((time,epochs))
    locals()[f"stime_smmlp_{model}"] = np.zeros((time,epochs))
    # precision_score_train = np.zeros((time,epochs))
    # recall_score_train = np.zeros((time,epochs))
    locals()[f"f1_score_train_smmlp_{model}"] = np.zeros((time,epochs))
    locals()[f"Accuracy_train_smmlp_{model}"] = np.zeros((time,epochs))
    locals()[f"kappa_train_smmlp_{model}"] = np.zeros((time,epochs))
    locals()[f"mse_train_smmlp_{model}"] = np.zeros((time,epochs))
    # precision_score = np.zeros((time,epochs))
    # recall_score = np.zeros((time,epochs))
    locals()[f"f1_score_test_smmlp_{model}"] = np.zeros((time,epochs))
    locals()[f"Accuracy_test_smmlp_{model}"] = np.zeros((time,epochs))
    locals()[f"kappa_test_smmlp_{model}"] = np.zeros((time,epochs))
    locals()[f"mse_test_smmlp_{model}"] = np.zeros((time,epochs))
    
    globals()[f"erro_{model}"] = np.zeros((time,epochs))
    globals()[f"accuracyy_{model}"] = np.zeros((time,epochs))
    
    locals()[f"accuracy_{model}"] = np.zeros((time,epochs))
    locals()[f"accuracy_test_{model}"] = np.zeros((time,epochs))
    locals()[f"loss_r_train_{model}"] = np.zeros((time,epochs))
    locals()[f"loss_r_test_{model}"] = np.zeros((time,epochs))
    locals()[f"std_accuracy_train_{model}"] = np.zeros((time))
    locals()[f"std_accuracy_test_{model}"] = np.zeros((time))
    locals()[f"std_loss_train_{model}"] = np.zeros((time))
    locals()[f"std_loss_test_{model}"] = np.zeros((time))
    globals()[f"lr_s_{model}"] = np.zeros((time,epochs))
    globals()[f"update_{model}"] = np.zeros((time))
    locals()[f"n_update_{model}"] = np.zeros((time)) 
    locals()[f"n_zeros_{model}"] = np.zeros((time))

IL = 0
SL = 1

for times in range(time): 

    for model in models:
    
        if model == "mse":
            Gamma = np.linspace(0.0490, 0.0490, num=time)
        if model == "mlp":
            Gamma = np.linspace(0.5, 0.5, num=time)
        if model == "mvs":
            Gamma = np.linspace(0.5576, 0.5576, num=time)
        if model == "mvsa":
            Gamma = np.linspace(0.2065, 0.2065, num=time)
        if model == "vs":
            Gamma = np.linspace(0.2609, 0.2609, num=time)
        if model == "vsa":
            Gamma = np.linspace(0.0369, 0.0369, num=time)  
        if model == "sm":
            Gamma = np.linspace(0.1459, 0.1459, num=time) 
        if model == "smnn":
            Gamma = np.linspace(0.0611, 0.0611, num=time)             
        
        if model == "emse":
            Gamma = np.linspace(0.2731, 0.2731, num=time)
            IL = 0.007
            SL = 0.0099
        if model == "emvs":
            Gamma = np.linspace(0.3154, 0.3154, num=time)
            IL = 0.0005
            SL = 0.032
        if model == "emvsa":
            Gamma = np.linspace(0.4244, 0.4244, num=time)
            IL = 0.0404
            SL = 0.11
        if model == "evs":
            Gamma = np.linspace(0.2731, 0.2731, num=time)
            IL = 0.004
            SL = 0.016
        if model == "evsa":
            Gamma = np.linspace(0.3275, 0.3275, num=time)  
            IL = 0.0462
            SL = 0.114
        if model == "esm":
            Gamma = np.linspace(0.3820, 0.3820, num=time) 
            IL = 0.0429
            SL = 0.0999
        if model == "esmnn":
            Gamma = np.linspace(0.5334, 0.5334, num=time) 
            IL = 0.0015
            SL = 0.0025
                


        # # define MLP model
        # model3 = Model_MLP()
        
        # LEARNING_RATE = 0.001
        
        # # compile the model
        # # model3.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        
            
        # model3.compile(optimizer=keras.optimizers.Adam(0.1),
        #             # loss=tf.keras.losses.MeanSquaredError(),
        #              loss=tf.keras.losses.MeanAbsoluteError(),
        #             # loss=my_loss_fn,
        #               metrics=[tf.keras.metrics.BinaryAccuracy()])
        
        
        ## Trying to get weights from MLP
        
        # a = np.array(model3.get_weights())         # save weights in a np.array of np.arrays
        # model.set_weights(a + 1)                  # add 1 to all weights in the neural network
        # b = np.array(model.get_weights())         # save weights a second time in a np.array of np.arrays
        # print(b - a)                              # print changes in weights    
        # model3.layers[0].get_weights()[0]
    
        # model.compile(loss=my_loss_fn,
        #               optimizer='adam',
        #               metrics=['accuracy'])
        
        batch_size = 100
        
        # d = load(file)
        # # ------------------------------------------------------- #
        # # remove zero rings
        # m_rings = m_rings = list(range(8,80)) + list(range(88,100))
        # data = norm1(d['data'][:,m_rings])
        # # ------------------------------------------------------- #
        # target = d['target']
        # #target[target!=1]=-1
        # cv = StratifiedKFold(n_splits=10, random_state=512, shuffle=True)
        # splits = [(train_index, val_index) for train_index, val_index in cv.split(data,target)]
        
        base = []
        base = mat5['x']
        
        # base = np.concatenate((base, base), axis=0)
        
        data = base[:,0:(base.shape[1] -1)]
        target = base[:,base.shape[1] -1]
        target[target==-1]=0
        
        #     #########################################################
        # Convert the nominal y values to binary
        
        data = normalize(data)
        target = target.astype(int)
        target = to_categorical(target)
        
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        
        #########################################################
        
        n_splits = 5
        # cv = KFold(n_splits=n_splits, random_state=1024, shuffle=True)
        # splits = [(train_index, val_index) for train_index, val_index in cv.split(data)]    
        cv = StratifiedKFold(n_splits=n_splits, random_state=1024, shuffle=True)
        splits = [(train_index, val_index) for train_index, val_index in cv.split(data,np.argmax(target, axis=1))]
        # x_train, x_val, y_train, y_val = \
        #     train_test_split(data, target, test_size=.2, random_state=42)
            

        sort = int(np.random.rand()*n_splits)
        x_train = data [ splits[sort][0]]
        y_train = target [ splits[sort][0] ]
        x_val = data [ splits[sort][1]]
        y_val = target [ splits[sort][1] ]
        # a = d['features']
        y_train = y_train.astype(np.float32)
        
        
        
        # scaler = preprocessing.MinMaxScaler()
        # scaler = preprocessing.Normalizer()
        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.fit_transform(x_val)
        
        
        
        # # learning schedule callback
        # loss_history = LossHistory()
        # LEARNING_RATE = LearningRateScheduler(step_decay)
        # callbacks_list = [loss_history, LEARNING_RATE]
        
        # # fit the model
        # history3 = model3.fit(x_train, y_train, 
        #                       validation_data=(x_val, y_val), 
        #                       epochs=epochs, 
        #                       batch_size=batch_size, 
        #                       callbacks=callbacks_list, 
        #                       verbose=1)        
        
        # # history3 = model3.fit(x_train, y_train, epochs=100, batch_size=100, verbose=1)
        # y_pred1_train_smmlp = model3.predict(x_train)
        # y_pred_train_smmlp = np.argmax(y_pred1_train_smmlp, axis=1)
        # y_pred1_smmlp = model3.predict(x_val)
        # y_pred_smmlp = np.argmax(y_pred1_smmlp, axis=1)    
    
        
        # MLP
        start = timee.time()
        clf = MultilayerPerceptron(n_hidden=7, n_iterations=epochs, learning_rate=0.002, model=model)
        
        # clf.fit(X_train, y_train)
        clf.fit(x_train, y_train)
        
        #########################################################
            
        # y_pred = np.argmax(clf.predict(X_test), axis=1)
        # y_test = np.argmax(y_test, axis=1)
        
        # y_pred = np.argmax(clf.predict(x_val), axis=1)
        # y_test = np.argmax(y_val, axis=1)
    
        y_pred_train_smmlp = np.argmax(clf.predict(x_train), axis=1)
        y_pred_smmlp = np.argmax(clf.predict(x_val), axis=1)
        end = timee.time() 
        y_trainn = np.argmax(y_train, axis=1)
        y_testt = np.argmax(y_val, axis=1)
        locals()[f"stime_smmlp_{model}"][times] = (end-start)
        
        #########################################################
        
        locals()[f"accuracy_{model}"][times] = accuracy_score(y_trainn, y_pred_train_smmlp)
        print ("Accuracy Train:", locals()[f"accuracy_{model}"][times,1])
        
        locals()[f"accuracy_test_{model}"][times] = accuracy_score(y_testt, y_pred_smmlp)
        print ("Accuracy Test:", locals()[f"accuracy_test_{model}"][times,1])    
        
        #########################################################
        
        print("Gamma :", Gamma[times], "IL :", IL, "SL :", SL)
        print("Time :", times) 

        print("<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>")
        
            # Print f1, precision, and recall scores for train
        # precision_score_train = precision_score(y_train, y_pred_train, average="macro")
        # recall_score_train = recall_score(y_train, y_pred_train, average="macro")
        locals()[f"kappa_train_smmlp_{model}"][times] = cohen_kappa_score(y_trainn, y_pred_train_smmlp)
        locals()[f"f1_score_train_smmlp_{model}"][times] = f1_score(y_trainn, y_pred_train_smmlp, average="macro")
        locals()[f"mse_train_smmlp_{model}"][times] = mean_squared_error(y_trainn, y_pred_train_smmlp)     
        locals()[f"Accuracy_train_smmlp_{model}"][times]  = accuracy_score(y_trainn, y_pred_train_smmlp)
        
            # Print f1, precision, and recall scores
        # precision_score = precision_score(y_val, y_pred, average="macro")
        # recall_score = recall_score(y_val, y_pred, average="macro")
        locals()[f"kappa_test_smmlp_{model}"][times] = cohen_kappa_score(y_testt, y_pred_smmlp)
        locals()[f"f1_score_test_smmlp_{model}"][times] = f1_score(y_testt, y_pred_smmlp, average="macro")
        locals()[f"mse_test_smmlp_{model}"][times] = mean_squared_error(y_testt, y_pred_smmlp)   
        locals()[f"Accuracy_test_smmlp_{model}"][times]  = accuracy_score(y_testt, y_pred_smmlp)
        
        # scores = model3.evaluate(x_val, y_val, verbose=1)
        # print("Final test loss and accuracy :", scores)
        
        # accuracy[times] = history3.history['binary_accuracy']
        # accuracy_test[times] = history3.history['val_binary_accuracy']
        locals()[f"mean_accuracy_train_{model}"] = np.mean(locals()[f"accuracy_{model}"])
        locals()[f"mean_accuracy_test_{model}"] = np.mean(locals()[f"accuracy_test_{model}"])
        # loss_r_train[times] = history3.history['loss'] 
        # loss_r_test[times] = history3.history['val_loss'] 
        locals()[f"mean_loss_train_{model}"] = np.mean(locals()[f"loss_r_train_{model}"], axis=1)
        locals()[f"mean_loss_test_{model}"] = np.mean(locals()[f"loss_r_test_{model}"], axis=1)
        locals()[f"std_accuracy_train_{model}"] = np.std(locals()[f"accuracy_{model}"], axis=1)
        locals()[f"std_accuracy_test_{model}"] = np.std(locals()[f"accuracy_test_{model}"], axis=1)
        locals()[f"std_loss_train_{model}"] = np.std(locals()[f"loss_r_train_{model}"], axis=1)
        locals()[f"std_loss_test_{model}"] = np.std(locals()[f"loss_r_test_{model}"], axis=1)
        # lr_s[times] = history3.history['lr']
        # update[times] = update
        # n_update[times] = n_update
        
        locals()[f"n_zeros_{model}"] = np.count_nonzero(globals()[f"lr_s_{model}"]==0,1)
        locals()[f"redcomplex_{model}"] = locals()[f"n_zeros_{model}"]/epochs

    
        print(model)
     
        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                        "Naive Bayes", "QDA"]
                                      

        # names = ["Linear SVM"]

        
        n_a=1
        
    
        
        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.2, max_iter=1000),
            # SVC(kernel="linear", C=0.025, max_iter=1000),
            SVC(gamma=2, C=1, max_iter=1000),
           # GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000, learning_rate_init=0.002, hidden_layer_sizes=7, 
            activation="logistic", solver="sgd"),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]
    
    
        k = 0
        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            start = timee.time()
            clf.fit(x_train, y_trainn)
            score = clf.score(x_val, y_testt)
            y_pred = clf.predict(x_val)
            y_pred_train = clf.predict(x_train)
            end = timee.time() 
            kappa[times + time * models.index(model),k] = cohen_kappa_score(y_testt, y_pred)
            fscore[times + time * models.index(model),k] = f1_score(y_testt, y_pred, average='macro')
            mse[times + time * models.index(model),k] = mean_squared_error(y_testt, y_pred)        
            Accuracy[times + time * models.index(model),k] = accuracy_score(y_testt, y_pred)
            kappa_train[times + time * models.index(model),k] = cohen_kappa_score(y_trainn, y_pred_train)
            fscore_train[times + time * models.index(model),k] = f1_score(y_trainn, y_pred_train, average='macro')
            mse_train[times + time * models.index(model),k] = mean_squared_error(y_trainn, y_pred_train)        
            Accuracy_train[times + time * models.index(model),k]  = accuracy_score(y_trainn, y_pred_train)
            stime_scikit[times + time * models.index(model),k]=(end-start)
            # results2[k,times] = (score*100)
            k +=1      
    # SMMLP['accuracy_test'] = accuracy_test
    # SMMLP['mean_accuracy_train'] = mean_accuracy_train

    # matrix.accuracy_test = accuracy_test
    # matrix.mean_accuracy_train = mean_accuracy_train
    # matrixA["SMMLP_{}".format(model)]['acurracy_train_{}'.format(model)] = accuracy
    
    # vars = ['accuracy', 'mean_accuracy_train', 'mean_accuracy_test']

        locals()[f"SMMLP_{model}"] = {'stime_smmlp': locals()[f"stime_smmlp_{model}"].copy(),
                                      'accuracy_test': locals()[f"accuracy_test_{model}"].copy(),
                                      'mean_accuracy_train': locals()[f"mean_accuracy_train_{model}"].copy(), 
                                      'mean_accuracy_test': locals()[f"mean_accuracy_test_{model}"].copy(),                                     
                                      'loss_r_train': locals()[f"loss_r_train_{model}"].copy(),
                                      'loss_r_test': locals()[f"loss_r_test_{model}"].copy(),
                                      'mean_loss_train': locals()[f"mean_loss_train_{model}"].copy(),
                                      'mean_loss_test': locals()[f"mean_loss_test_{model}"].copy(),
                                      'std_accuracy_train': locals()[f"std_accuracy_train_{model}"].copy(),
                                      'std_accuracy_test': locals()[f"std_accuracy_test_{model}"].copy(),
                                      'std_loss_train': locals()[f"std_loss_train_{model}"].copy(),
                                      'std_loss_test': locals()[f"std_loss_test_{model}"].copy(),
                                      'lr_s': globals()[f"lr_s_{model}"].copy(),    
                                      'update': globals()[f"update_{model}"].copy(),
                                      'n_update': locals()[f"n_update_{model}"].copy(),
                                      'Gamma': Gamma.copy(),
                                      'f1_score_train_smmlp': locals()[f"f1_score_train_smmlp_{model}"].copy(),
                                      'f1_score_test_smmlp': locals()[f"f1_score_test_smmlp_{model}"].copy(),
                                      'kappa_train_smmlp': locals()[f"kappa_train_smmlp_{model}"].copy(),
                                      'kappa_test_smmlp': locals()[f"kappa_test_smmlp_{model}"].copy(),
                                      'Accuracy_train_smmlp': locals()[f"Accuracy_train_smmlp_{model}"].copy(),
                                      'Accuracy_test_smmlp': locals()[f"Accuracy_test_smmlp_{model}"].copy(),
                                      'Erro': locals()[f"erro_{model}"].copy(),
                                      'n_zeros': locals()[f"n_zeros_{model}"].copy(),
                                      'redcomplex': locals()[f"redcomplex_{model}"].copy(),
                                      'accuracyy' : globals()[f"accuracyy_{model}"].copy()                                                                                                                             
                                      }
        # print(SMMLP_mse['f1_score_train_smmlp']) 
        matrixA[f"SMMLP_{model}"] = eval(f"SMMLP_{model}")
    # setattr(matrixA, f"SMMLP_{model}", eval(f"SMMLP_{model}"))
    # matrix.(f"SMMLP_{model}") = [f"SMMLP_{model}"]
    # for v in vars: 
    #     setattr(matrix, v, eval(v)) 
    
    # matrixA["SMMLP_{}".format(model)]['accuracy_test'] = accuracy_test
    # matrixA["SMMLP_{}".format(model)]['mean_accuracy_train'] = mean_accuracy_train
    # matrixA["SMMLP_{}".format(model)]['mean_accuracy_test'] = mean_accuracy_test
    # matrixA["SMMLP_{}".format(model)]['loss_r_train'] = loss_r_train
    # matrixA["SMMLP_{}".format(model)]['loss_r_test'] = loss_r_test
    # matrixA["SMMLP_{}".format(model)]['mean_loss_train'] = mean_loss_train
    # matrixA["SMMLP_{}".format(model)]['mean_loss_test'] = mean_loss_test
    # matrixA["SMMLP_{}".format(model)]['std_accuracy_train'] = std_accuracy_train
    # matrixA["SMMLP_{}".format(model)]['std_accuracy_test'] = std_accuracy_test
    # matrixA["SMMLP_{}".format(model)]['std_loss_train'] = std_loss_train
    # matrixA["SMMLP_{}".format(model)]['std_loss_test'] = std_loss_test
    # matrixA["SMMLP_{}".format(model)]['lr_s'] = lr_s    
    # matrixA["SMMLP_{}".format(model)]['update'] = update
    # matrixA["SMMLP_{}".format(model)]['n_update'] = n_update
    # matrixA["SMMLP_{}".format(model)]['Gamma'] = Gamma
    # matrixA["SMMLP_{}".format(model)]['f1_score_train_smmlp'] = f1_score_train_smmlp
    # matrixA["SMMLP_{}".format(model)]['f1_score_test_smmlp'] = f1_score_test_smmlp
    # matrixA["SMMLP_{}".format(model)]['kappa_train_smmlp'] = kappa_train_smmlp
    # matrixA["SMMLP_{}".format(model)]['kappa_test_smmlp'] = kappa_test_smmlp
    # matrixA["SMMLP_{}".format(model)]['Accuracy_train_smmlp'] = Accuracy_train_smmlp
    # matrixA["SMMLP_{}".format(model)]['Accuracy_test_smmlp'] = Accuracy_test_smmlp
    # matrixA["SMMLP_{}".format(model)]['Erro'] = erro            


mean_Accuracy = np.mean(Accuracy,0)
mean_Accuracy_train = np.mean(Accuracy_train,0)
std_Accuracy = np.std(Accuracy,0)  
std_Accuracy_train = np.std(Accuracy_train,0)

mean_mse = np.mean(Accuracy,0)  
mean_mse_train = np.mean(Accuracy_train,0)
std_mse = np.std(Accuracy,0)  
std_mse_train = np.std(Accuracy_train,0)



    
    
    # matrixA['SMMLP']['acurracy_train'] = accuracy
    # matrixA['SMMLP']['accuracy_test'] = accuracy_test
    # matrixA['SMMLP']['mean_accuracy_train'] = mean_accuracy_train
    # matrixA['SMMLP']['mean_accuracy_test'] = mean_accuracy_test
    # matrixA['SMMLP']['loss_r_train'] = loss_r_train
    # matrixA['SMMLP']['loss_r_test'] = loss_r_test
    # matrixA['SMMLP']['mean_loss_train'] = mean_loss_train
    # matrixA['SMMLP']['mean_loss_test'] = mean_loss_test
    # matrixA['SMMLP']['std_accuracy_train'] = std_accuracy_train
    # matrixA['SMMLP']['std_accuracy_test'] = std_accuracy_test
    # matrixA['SMMLP']['std_loss_train'] = std_loss_train
    # matrixA['SMMLP']['std_loss_test'] = std_loss_test
    # matrixA['SMMLP']['lr_s'] = lr_s    
    # matrixA['SMMLP']['update'] = update
    # matrixA['SMMLP']['n_update'] = n_update
    # matrixA['SMMLP']['Gamma'] = Gamma
    # matrixA['SMMLP']['f1_score_train_smmlp'] = f1_score_train_smmlp
    # matrixA['SMMLP']['f1_score_test_smmlp'] = f1_score_test_smmlp
    # matrixA['SMMLP']['kappa_train_smmlp'] = kappa_train_smmlp
    # matrixA['SMMLP']['kappa_test_smmlp'] = kappa_test_smmlp
    # matrixA['SMMLP']['Accuracy_train_smmlp'] = Accuracy_train_smmlp
    # matrixA['SMMLP']['Accuracy_test_smmlp'] = Accuracy_test_smmlp
    # matrixA['SMMLP']['Erro'] = erro

  
Scikit = {'kappa' : kappa,
          'fscore' : fscore,
          'mse' : mse ,
          'Accuracy' : Accuracy ,
          'mean_Accuracy' : mean_Accuracy ,
          'kappa_train' : kappa_train ,
          'fscore_train' : fscore_train ,
          'mse_train' : mse_train ,
          'Accuracy_train' : Accuracy_train ,
          'mean_Accuracy_train' : mean_Accuracy_train ,
          'std_Accuracy' : std_Accuracy ,
          'std_Accuracy_train' : std_Accuracy_train ,
          'mean_mse' : mean_mse ,
          'mean_mse_train' : mean_mse_train ,
          'std_mse' : std_mse ,
          'stime_scikit' : stime_scikit ,
          'std_mse_train' : std_mse_train}
# setattr(matrixA, 'Scikit', eval('Scikit'))
matrixA["Sciki"] = eval("Scikit")



savemat("matlab_matrix_220406.mat", matrixA)
print(matrixA)
