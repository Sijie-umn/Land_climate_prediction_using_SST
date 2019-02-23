import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from math import sqrt
import math
import numpy as np
import copy
import os,sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import properties
from scipy.io import loadmat
from lcp_utility import utility
import datetime as dt
from scipy.stats.stats import pearsonr

import keras
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
import keras.callbacks as cllbck
from keras import regularizers

import xgboost as xgb


class nonlinear_mlp(object):
    def __init__(self):
        # Load data
        self.dir=properties.DATA_PATH
        self.x_data_file=properties.FILE_X
        self.y_data_file=properties.FILE_Y
        self.result_path = properties.RES_PATH
        self.result_file = properties.RES_FILE_MLP
        # neural network parameters
        self.epoch = properties.NO_EPOCH
        self.layers = properties.NO_LAYERS
        self.hidden_units = properties.NO_HIDDEN_UNITS
        self.batch = properties.BATCH_SIZE
        self.optimizer = properties.OPTIMIZER
        self.regularizer = properties.REGULARIZER
        self.early_stop = properties.EARLY_STOP
        self.learning_rate_decay = properties.LEARNING_RATE_DECAY
        ############################
        self.nb_models = properties.NO_MODELS
        self.nb_locations = properties.NO_LOC


    def build_mlp(self,input_shape, nb_layers = 1, nb_hidden_units = 2, optimizer = 'Adam', regularizer = 'None',
                  weight_l1 = 0.01, weight_l2 = 0.01, weight_l1_l2 = 0.01):
    
        """
        Build 1-layer MLP model
        input_shape: shape of input images
        nb_hidden_units: number of hidden units at each hidden layer
        optimizer: Optimization algorithm for Gradient decent:vanilla SGD, Momentum, RMSprop, Adam
        regularizer: A choice from L1, L2 and L1_L2 regulariztion
        weight_l1: weight for l1 regularizer, default is 0.01
        weight_l2: weight for l2 regularizer, default is 0.01
        weight_l1_l2: weight for l1_l2 regularizer, default is 0.01

        Return:
        MLP model
        """

        # Add regularization
        if regularizer == 'L1':
            regularizer = regularizers.l1(weight_l1)
        elif regularizer == 'L2':
            regularizer = regularizers.l2(weight_l2)
        elif regularizer == 'L1_l2':
            regularizer = regularizers.l1_l2(weight_l1_l2)
        else:
            regularizer = regularizers.l1(0.)

        # Build Multi-layer perceptron model
        mlp = Sequential()
        mlp.add(Dense(nb_hidden_units,input_dim = input_shape,activation='relu',kernel_regularizer=regularizer))

        if nb_layers > 1:
            for l in range(2,nb_layers+1):
                mlp.add(Dense(nb_hidden_units,activation='relu',kernel_regularizer = regularizer))
        mlp.add(Dense(1, activation = None,name = 'prediction'))

        mlp.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['mse'])
        print(mlp.summary())

        return mlp
    


    def fit_mlp(self,X,y,mlp,batch_size=32,early_stop = False,lr_decay = False,nb_epochs=100):

        """Train and evaluate CNN.
        Args:
            x_train: input data matrix (images) for training
            y_train: labels associated with training data matrix


        mlp: multi-layer perceptron model

            nb_epochs: number of training epochs
            batch_size: number of samples in the batch
            early_stop: Boolean, decide if we will use early stopping
            lr_decay: Boolean, decide wether we will perform learning rate decay
            history_log: training and validation history


        Return:
            trained MLP model
        """


        # Add callbacks
        monitor = 'val_loss'


        #csv_logger = cllbck.CSVLogger(log_name)
        callbacks_list = []

        if early_stop:
            early_stopping = cllbck.EarlyStopping(monitor=monitor, patience=10, verbose=1)
            callbacks_list.append(early_stopping)
        if lr_decay:
            reduce_lr = cllbck.ReduceLROnPlateau(monitor=monitor, factor=0.2,patience=10, min_lr=0.0001)
            callbacks_list.append(reduce_lr)

        data_len = len(y)
        train_len = data_len - 20*12
        x_train, x_val, y_train, y_val = X[:train_len,:], X[train_len:,:], y[:train_len], y[train_len:]
        history = mlp.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=nb_epochs,
                          callbacks=callbacks_list,
                          validation_data=(x_val, y_val))
        return mlp


    def predict_mlp(self,mlp,X):
        y_pred = mlp.predict(X)
        return y_pred
                          



    def run_mlp(self):
        # Load training and test datasets
        X=pickle.load(open(os.path.join(self.dir,self.x_data_file),'rb'))# 3 models, 5 sets, 1:train + 1:test
        Y=pickle.load(open(os.path.join(self.dir,self.y_data_file),'rb'))# 3 models, 5 sets, 1:train + 1:test, 3 locations
        Y_pre_train_mlp = []
        Y_pre_test_mlp = []



        input_shape = X[0][0][0].shape[1]
        test_length = Y[0][0][1].shape[0] # length of test data set: 120
        train_length = Y[0][0][0].shape[0] - 12*20 # length of train data :1200
        l_train = Y[0][0][0].shape[0]


        #for m in range(self.nb_models):
        for climate_model in range(self.nb_models):
            nb_windows = len(X[climate_model]) # number of shifting windows 5 or 6 this need to be decided later
            target_train_mlp = []
            target_test_mlp = []

            for loc in range(self.nb_locations):
                temp_train_mlp = []
                temp_test_mlp = []
                for window in range(nb_windows): # s change from 1 to 5/6

                    X_train = copy.deepcopy(X[climate_model][window][0])
                    X_test = copy.deepcopy(X[climate_model][window][1])
                    Y_train = copy.deepcopy(Y[climate_model][window][0][:, loc])
                    Y_test = copy.deepcopy(Y[climate_model][window][1][:, loc])

                    mlp = self.build_mlp(input_shape, nb_layers = self.layers, nb_hidden_units = self.hidden_units, optimizer = self.optimizer, regularizer = self.regularizer)
                    mlp = self.fit_mlp(X_train, Y_train, mlp,batch_size=self.batch, early_stop = self.early_stop,lr_decay = self.learning_rate_decay,nb_epochs=self.epoch)


                    pre_train=self.predict_mlp(mlp,X_train)
                    pre_test=self.predict_mlp(mlp,X_test)
                    temp_train_mlp.append(pre_train)
                    temp_test_mlp.append(pre_test)
                target_train_mlp.append(temp_train_mlp)
                target_test_mlp.append(temp_test_mlp)

            Y_pre_train_mlp.append(target_train_mlp)
            Y_pre_test_mlp.append(target_test_mlp)
            result=[Y_pre_train_mlp,Y_pre_test_mlp]

        file=os.path.join(self.result_path, self.result_file)

        with open(file, 'wb') as f:
                pickle.dump(result, f)
        



       
      
class nonlinear_gbt(object):
    def __init__(self):
        # Load data
        self.dir=properties.DATA_PATH
        self.x_data_file=properties.FILE_X
        self.y_data_file=properties.FILE_Y
        self.result_path = properties.RES_PATH
        self.result_file = properties.RES_FILE_GBT
        

        self.round = properties.NO_ROUND
        self.eta = properties.ETA
        self.depth = properties.MAX_DEPTH
        self.eval_metric = properties.EVAL_METRIC

        ############################
        self.nb_models = properties.NO_MODELS
        self.nb_locations = properties.NO_LOC                          




    def fit_gbt(self,param, X,y, num_round=100, early_stopping = False):
                          
        """
        Build Gradient Boosting Tree model
        Parameters:
            eta:  learning_rate, [default=0.3 ] grid search [2 to 10]/num_round
            max_depth: maximum depth of a tree, [default=6] Grid search [3 to 10]
            num_round: aka, number of boosted trees, keep it around 100

            lambda: L2 regularization term on weights, [default=1] increase this value will make model more conservative.
            alpha: L1 regularization term on weights, [default=0] increase this value will make model more conservative.
            gamma: minimum loss reduction required to make a further partition on a leaf node of the tree, [default=0, alias: min_split_loss] keep it as 0

        Return:
            GBT model

        """
        data_len = len(y)
        train_len = data_len - 20*12
        x_train, x_val, y_train, y_val = X[:train_len,:], X[train_len:,:], y[:train_len], y[train_len:]

        dtrain = xgb.DMatrix(x_train,label=y_train)
        #dtest = xgb.DMatrix(x_test,label=y_test)
        dval = xgb.DMatrix(x_val, label = y_val)
        evallist  = [(dval,'Val'), (dtrain,'train')]
                  
        if early_stopping:
            bst = xgb.train(param, dtrain, num_round,evallist,early_stopping_rounds=10)
        else:
            bst = xgb.train(param, dtrain, num_round,evallist)

        return bst


    def predict_gbt(self,bst,X,y):
        
        data = xgb.DMatrix(X,label=y)
        y_pred = bst.predict(data)

        return y_pred

       

    def run_gbt(self):
        X=pickle.load(open(os.path.join(self.dir,self.x_data_file),'rb'))
        Y=pickle.load(open(os.path.join(self.dir,self.y_data_file),'rb'))
        Y_pre_train_gbt = []
        Y_pre_test_gbt= []

        for climate_model in range(self.nb_models):
            nb_windows = len(X[climate_model]) # number of shifting windows 5 or 6 this need to be decided later
            target_train_gbt = []
            target_test_gbt = []

            for loc in range(self.nb_locations):
                temp_train_gbt= []
                temp_test_gbt = []
                for window in range(nb_windows): # s change from 1 to 5/6

                    X_train = copy.deepcopy(X[climate_model][window][0])
                    X_test = copy.deepcopy(X[climate_model][window][1])
                    Y_train = copy.deepcopy(Y[climate_model][window][0][:, loc])
                    Y_test = copy.deepcopy(Y[climate_model][window][1][:, loc])

                    param = {'max_depth':self.depth, 'eta':self.eta/self.round,'silent':1, 'objective':'reg:linear' }
                    param['nthread'] = 4
                    param['eval_metric'] =  self.eval_metric

                    bst= self.fit_gbt(param, X_train,Y_train, num_round=self.round, early_stopping = False)

                    pre_train=self.predict_gbt(bst,X_train,Y_train)
                    pre_test=self.predict_gbt(bst,X_test, Y_test)
                    temp_train_gbt.append(pre_train)
                    temp_test_gbt.append(pre_test)
                target_train_gbt.append(temp_train_gbt)
                target_test_gbt.append(temp_test_gbt)

            Y_pre_train_gbt.append(target_train_gbt)
            Y_pre_test_gbt.append(target_test_gbt)
            result=[Y_pre_train_gbt,Y_pre_test_gbt]

        utility.save_results(self.result_path, self.result_file,result)

        #file=os.path.join(self.result_path, self.result_file)

        #with open(file, 'wb') as f:
                #pickle.dump(result, f)
                

