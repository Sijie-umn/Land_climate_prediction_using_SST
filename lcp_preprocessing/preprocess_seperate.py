from scipy.io import loadmat
import numpy as np
import math as m
import pickle
import copy
import properties
import os

def preprocess(X_train, X_test, Y_train, Y_test):
    [tr_sample, num_feature] = X_train.shape
    [te_sample, num_feature] = X_test.shape
    [te_sample,num_target]=Y_test.shape

    model_X = []
    model_Y = []
    month_X = []
    mean_X = []
    std_X = []
    month_Y = []
    mean_Y = []
    std_Y = []

    for j in range(12):
        temp = []
        month_X.append(temp)
        mean_X.append(temp)
        std_X.append(temp)
        month_Y.append(temp)
        mean_Y.append(temp)
        std_Y.append(temp)
    for j in range(tr_sample):
        temp1 = j % 12
        if j < 12:
            month_X[temp1] = X_train[j]
            month_Y[temp1] = Y_train[j]
        else:
            month_X[temp1] = np.vstack([month_X[temp1], X_train[j]])
            month_Y[temp1] = np.vstack([month_Y[temp1], Y_train[j]])
    for j in range(12):
        mean_X[j] = np.mean(month_X[j], axis=0)
        std_X[j] = np.std(month_X[j], axis=0)
        mean_Y[j] = np.mean(month_Y[j], axis=0)
        std_Y[j] = np.std(month_Y[j], axis=0)

    for j in range(tr_sample):
        temp1 = j % 12
        X_train[j] = np.subtract(X_train[j], mean_X[temp1])
        Y_train[j] = np.subtract(Y_train[j], mean_Y[temp1])
        for k in range(num_feature):
            if std_X[temp1][k] != 0:
                X_train[j][k] = X_train[j][k] / std_X[temp1][k]
            else:
                print('std=0')
        for k in range(num_target):
            if std_Y[temp1][k] != 0:
                Y_train[j][k] = Y_train[j][k] / std_Y[temp1][k]
            else:
                print('std=0')
    for j in range(te_sample):
        temp1 = j % 12
        X_test[j] = np.subtract(X_test[j], mean_X[temp1])
        Y_test[j] = np.subtract(Y_test[j], mean_Y[temp1])
        for k in range(num_feature):
            if std_X[temp1][k] != 0:
                X_test[j][k] = X_test[j][k] / std_X[temp1][k]
            else:
                print('std=0')
        for k in range(num_target):
            if std_Y[temp1][k] != 0:
                Y_test[j][k] = Y_test[j][k] / std_Y[temp1][k]
            else:
                print('std=0')
    model_X.append(X_train)
    model_X.append(X_test)

    model_Y.append(Y_train)
    model_Y.append(Y_test)
    return (model_X,model_Y)

os.chdir(properties.Data_path)

name_mat=['X_tos_2.5x2.5.mat']#SST file name
name_file=['X_sep5_2x2.p']#Preprocessed SST file name
for name in range(len(name_mat)):
    data=loadmat(name_mat[name])
    X=data['X_tos']
    data=loadmat('y_tas.mat')
    Y=data['y_tas']
    Model_index=np.array([[0,1872],[12960,14832],[14832,16788]])
    [num_model,a]=Model_index.shape
    X_model=[]
    Y_model=[]
    for i in range(num_model):
        X_data=copy.deepcopy(X[Model_index[i][0]:Model_index[i][1]])
        Y_data=copy.deepcopy(Y[Model_index[i][0]:Model_index[i][1]])
        Y_data=Y_data[:,(0,3,6)]
        [num_sample,num_feature]=X_data.shape
        train_years=properties.Train_size
        test_years=properties.Test_size
        step_size=properties.Step_size
        num_years=num_sample/12
        if test_years<=step_size:
            num_window=m.floor(float(num_years-train_years)/step_size)
        else:
            num_window=m.floor(float(num_years-train_years-(test_years-step_size))/step_size)
        temp_X = []
        temp_Y = []
        for j in range(int(num_window)):
            X_Train=copy.deepcopy(X_data[j * step_size*12:j * step_size*12 + 12*train_years])
            X_Test=copy.deepcopy(X_data[j*step_size*12+12*train_years:j*step_size*12+12*train_years+12*test_years])
            Y_Train=copy.deepcopy(Y_data[j * step_size*12:j * step_size*12 + 12*train_years])
            Y_Test=copy.deepcopy(Y_data[j*step_size*12+12*train_years:j*step_size*12+12*train_years+12*test_years])
            model_X,model_Y=preprocess(X_Train,X_Test,Y_Train,Y_Test)
            temp_X.append(model_X)
            temp_Y.append(model_Y)
        X_model.append(temp_X)
        Y_model.append(temp_Y)
    pickle.dump( X_model, open( name_file[name], "wb" ) )
    pickle.dump( Y_model, open( "Y_sep5.p", "wb" ) )

