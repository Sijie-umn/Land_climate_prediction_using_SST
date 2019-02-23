import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
from scipy import stats
import os
import pickle

def train_validation(X,Y,validation_percentage=0.2,start=False,end=True):
    [num_sample,num_feature]=X.shape
    ns=len(Y)
    if(num_sample!=ns):
        print('The dimensions of X and Y are not matched')
        quit()
    if(end==True):
        split_point=int(num_sample-np.floor(num_sample*validation_percentage))
        Train_X = X[0:split_point]
        Valid_X = X[split_point:num_sample]
        Train_Y = Y[0:split_point]
        Valid_Y = Y[split_point:num_sample]
    else:
        if(start==True):
            split_point=int(np.floor(num_sample*validation_percentage))
            Valid_X = X[0:split_point]
            Train_X = X[split_point:num_sample]
            Valid_Y = Y[0:split_point]
            Train_Y= Y[split_point:num_sample]
        else:
            print('The arguments start and end are missing')
            quit()

    Train=[Train_X,Train_Y]
    Valid=[Valid_X,Valid_Y]
    return(Train,Valid)


def compute_r2(True_y,Pre_y):
    return r2_score(True_y,Pre_y)

def compute_rmse(True_y,Pre_y):
    return sqrt(mean_squared_error(True_y,Pre_y))

def KS_test(y1,y2):
    temp=stats.ks_2samp(y1, y2)
    return temp.pvalue


def save_results(results_folder,filename, results):
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    filepath = os.path.join(results_folder, filename)
    with open(filepath, 'wb') as fh:
        pickle.dump(results, fh)

class evaluation(object):
    def __init__(self):
        self.result_file_path = r'/Users/sijiehe/Research/Summer 2018/Experiment_datasets'
        self.result_file_name="result_alasso.p"
        self.true_file_path = r'/Users/sijiehe/Research/Summer 2018/Experiment_datasets'
        self.true_file_name="Y.sep"
        self.climate_model_nums=3
        self.target_model_nums=3


    def run_evlauation(self):
        Pre_y=pickle.load(open(os.path.join(self.result_file_path,self.result_file_name),'rb'))
        True_y = pickle.load(open(os.path.join(self.true_file_path, self.true_file_name), 'rb'))
        result_rmse=np.zeros((2,self.climate_model_nums,self.target_model_nums))
        result_r2 = np.zeros((2, self.climate_model_nums, self.target_model_nums))
        for i in range(self.climate_model_nums):
            for j in range(self.target_model_nums):
                num_window = len(Pre_y[i])
                temp = np.zeros((2, num_window))
                for k in range(num_window):
                    temp[0][k]=compute_rmse(Pre_y[i][j][k],True_y[i][j][k])
                    temp[1][k] = compute_r2(Pre_y[i][j][k], True_y[i][j][k])
                result_rmse[0][i][j]=np.average(temp,axis=1)[0]
                result_r2[0][i][j]=np.average(temp,axis=1)[1]
                result_rmse[1][i][j]=stats.sem(temp,axis=1)[0]
                result_r2[1][i][j]=stats.sem(temp,axis=1)[1]
        result=[result_rmse,result_r2]
        return result



