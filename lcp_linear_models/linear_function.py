import pickle
from sklearn.linear_model import Lasso,LassoCV,LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
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
from scipy.stats.stats import pearsonr


class linear_lasso(object):
    def __init__(self):
        self.lambda_seq=np.linspace(properties.Lasso_lambda_start,properties.Lasso_lambda_end,properties.Lasso_num_lambda)
        self.lambda_lasso=properties.Lasso_lambda
        self.x_data_file=properties.Lasso_X_file
        self.y_data_file=properties.Lasso_Y_file
        self.dir=properties.Lasso_Data_path
        self.validation_split=properties.Lasso_split
        self.model_nums = properties.Lasso_model_num
        self.target_nums=properties.Lasso_target_num
        self.result_file=properties.Lasso_result_file
        self.result_path=properties.Lasso_Result_path

    def lasso_validation(self,X,Y,lambda_seq,validation_split):
        [Train,Validation]=utility.train_validation(X,Y,validation_percentage=validation_split)
        num_lambda=len(lambda_seq)
        mse=np.zeros(num_lambda)
        mse_train=np.zeros(num_lambda)
        for i in range(num_lambda):
            beta=self.fit_lasso_regression(Train[0],Train[1],lambda_seq[i])
            y_pre=self.predict_lasso_regression(Validation[0],beta)
            y_train=self.predict_lasso_regression(Train[0],beta)
            mse[i]=mean_squared_error(Validation[1],y_pre)
            mse_train[i]=mean_squared_error(Train[1],y_train)
        idx=np.argmin(mse)
#        print("validation:")
#        print(np.sqrt(mse[idx]))
#        print("train:")
#        print(np.sqrt(mse_train[idx]))
        return lambda_seq[idx]

    def fit_lasso_regression(self,X,Y,lasso_lambda):
        clf=Lasso(alpha=lasso_lambda,fit_intercept=False,normalize=False,max_iter=10000,tol=1e-5)
        clf.fit(X,Y)
        return clf.coef_

    def predict_lasso_regression(self,X,beta):
        return np.matmul(X,beta)

    def run_lasso(self):
        X=pickle.load(open(os.path.join(self.dir,self.x_data_file),'rb'))
        Y=pickle.load(open(os.path.join(self.dir,self.y_data_file),'rb'))
        Y_pre_train_lasso = []
        Y_pre_test_lasso = []
        beta_record=[]
        for climate_model in range(self.model_nums):
            num_window = len(X[climate_model])
            target_train_lasso = []
            target_test_lasso = []
            target_beta=[]
            for target in range(self.target_nums):
                temp_train_lasso = []
                temp_test_lasso = []
                temp_beta=[]
                for window in range(num_window):
                    X_train = copy.deepcopy(X[climate_model][window][0])
                    X_test = copy.deepcopy(X[climate_model][window][1])
                    Y_train = copy.deepcopy(Y[climate_model][window][0][:, target])
                    Y_test = copy.deepcopy(Y[climate_model][window][1][:, target])
                    best_lambda=self.lasso_validation(X_train,Y_train,self.lambda_seq,self.validation_split)
                    print(best_lambda)
                    beta=self.fit_lasso_regression(X_train,Y_train,best_lambda)
                    temp_beta.append(beta)
                    pre_train=self.predict_lasso_regression(X_train,beta)
                    pre_test=self.predict_lasso_regression(X_test,beta)
                    temp_train_lasso.append(pre_train)
                    temp_test_lasso.append(pre_test)
                target_train_lasso.append(temp_train_lasso)
                target_test_lasso.append(temp_test_lasso)
                target_beta.append(temp_beta)
            Y_pre_train_lasso.append(target_train_lasso)
            Y_pre_test_lasso.append(target_test_lasso)
            beta_record.append(target_beta)
            result=[Y_pre_train_lasso,Y_pre_test_lasso]

        file=os.path.join(self.result_path, self.result_file)

        with open(file, 'wb') as f:
                pickle.dump(result, f)
        file = os.path.join(self.result_path, 'beta_lasso_Brzail.p')
        with open(file,'wb') as f:
            pickle.dump(beta_record,f)
        return beta_record


class linear_pcr(object):
    def __init__(self):
        self.x_data_file=properties.PCR_X_file
        self.y_data_file=properties.PCR_Y_file
        self.dir=properties.PCR_Data_path
        self.validation_split=properties.PCR_split
        self.model_nums = properties.PCR_model_num
        self.target_nums=properties.PCR_target_num
        self.result_file=properties.PCR_result_file
        self.result_path=properties.PCR_Result_path
        self.variance_thre=properties.PCR_thre_var
        self.step_size=properties.PCR_step_size



    def PCR_validation(self,X,Y,variance_thre,step_size,validation_split):
        [Train,Validation]=utility.train_validation(X,Y,validation_percentage=validation_split)
        X_r=Train[0]
        Y_r=Train[1]
        X_v=Validation[0]
        Y_v=Validation[1]
        [Tr_sample,num_feature]=X_r.shape
        pca=PCA()
        X_reduced_tr=pca.fit_transform(scale(X_r))
        var=pca.explained_variance_ratio_
        n=len(X_reduced_tr)
        acc_var=0
        for i in range(num_feature):
            acc_var=acc_var+var[i]
            if acc_var>variance_thre:
                num_pc=i
                break

        num=np.ceil(num_pc/step_size)
        num_pc_list=np.zeros(int(num))
        mse=np.zeros(int(num))
        for i in range(int(num)):
            temp=int(num_pc-i*step_size)
            regr=LinearRegression()
            regr.fit(X_reduced_tr[:,:temp],Y_r)
            V_pre=self.PCR_predict(X_r,X_v,regr.coef_)
            mse[i]=utility.compute_rmse(Y_v,V_pre)
            num_pc_list[i]=temp

        val, idx = min((val, idx) for (idx, val) in enumerate(mse))
        return num_pc_list[idx]

    def PCR_fit(self,X,Y,best_num):
        pca=PCA()
        X_reduced=pca.fit_transform(scale(X))[:,:int(best_num)]
        regr=LinearRegression()
        regr.fit(X_reduced,Y)
        return regr.coef_

    def PCR_predict(self,X_train,X,beta):
        pca=PCA()
        best_num=len(beta)
        pca.fit_transform(scale(X_train))
        X_reduced=pca.transform(scale(X))[:,:int(best_num)]
        return np.matmul(X_reduced,beta)

    def run_PCR(self):
        X=pickle.load(open(os.path.join(self.dir,self.x_data_file),'rb'))
        Y=pickle.load(open(os.path.join(self.dir,self.y_data_file),'rb'))
        Y_pre_train_PCR = []
        Y_pre_test_PCR = []
        for climate_model in range(self.model_nums):
            num_window = len(X[climate_model])
            target_train_PCR = []
            target_test_PCR = []
            for target in range(self.target_nums):
                temp_train = []
                temp_test = []
                for window in range(num_window):
                    X_train = copy.deepcopy(X[climate_model][window][0])
                    X_test = copy.deepcopy(X[climate_model][window][1])
                    Y_train = copy.deepcopy(Y[climate_model][window][0][:, target])
                    Y_test = copy.deepcopy(Y[climate_model][window][1][:, target])
                    print(window)
                    best_num_pc=self.PCR_validation(X=X_train,Y=Y_train,variance_thre=self.variance_thre,step_size=self.step_size,validation_split=self.validation_split)
                    beta=self.PCR_fit(X_train,Y_train,best_num=best_num_pc)
                    pre_train=self.PCR_predict(X_train,X_train,beta)
                    pre_test=self.PCR_predict(X_train,X_test,beta)
                    temp_train.append(pre_train)
                    temp_test.append(pre_test)
                target_train_PCR.append(temp_train)
                target_test_PCR.append(temp_test)
            Y_pre_train_PCR.append(target_train_PCR)
            Y_pre_test_PCR.append(target_test_PCR)
            result=[Y_pre_train_PCR,Y_pre_test_PCR]

        file=os.path.join(self.result_path, self.result_file)

        with open(file, 'wb') as f:
                pickle.dump(result, f)
#return time_pcr

class linear_alasso(object):
    def __init__(self):
        self.lambda_seq=np.linspace(properties.ALasso_lambda_start,properties.ALasso_lambda_end,properties.ALasso_num_lambda)
        self.lambda_lasso=properties.ALasso_lambda
        self.x_data_file=properties.ALasso_X_file
        self.y_data_file=properties.ALasso_Y_file
        self.dir=properties.ALasso_Data_path
        self.validation_split=properties.ALasso_split
        self.model_nums = properties.ALasso_model_num
        self.target=properties.ALasso_target
        self.result_file=properties.ALasso_result_file
        self.result_path=properties.ALasso_Result_path

    def cal_dis(self,latitude1, longitude1, latitude2, longitude2):
        latitude1 = (math.pi / 180.0) * latitude1
        latitude2 = (math.pi / 180.0) * latitude2
        longitude1 = (math.pi / 180.0) * longitude1
        longitude2 = (math.pi / 180.0) * longitude2
        R = 20
        temp = math.sin(latitude1) * math.sin(latitude2) + \
               math.cos(latitude1) * math.cos(latitude2) * math.cos(longitude2 - longitude1)
        # if repr(temp)>1.0:
        #      temp = 1.0
        d = math.acos(temp) * R
        return d

    def alasso_validation(self,X,Y,lambda_seq,validation_split,target_name):
        [Train,Validation]=utility.train_validation(X,Y,validation_percentage=validation_split)
        dict = {'Brazil': np.array([-10.0, 310.0]), 'Peru': np.array([-5.75, 283.0]), 'Asia': np.array([-10.0, 137.0])}
        target_location=dict[target_name]
        resolution = {332: [10, 'lat_lon_10x10.mat'], 1257: [5, 'lat_lon_5x5.mat'], 5881: [2.5, 'lat_lon_2.5x2.5.mat']}
        X_r=Train[0]
        [Tr_sample,num_feature]=X_r.shape
        step=resolution[num_feature][0]
        data=loadmat(resolution[num_feature][1])
        position=data['lat_lon_data']
        [row, column] = position.shape
        lat_block = np.zeros(row)
        lon_block = np.zeros(column)
        for i in range(row):
            lat_block[i] = -90 + step * i
        for i in range(column):
            lon_block[i] = 0 + i * step
        distance = np.zeros((row, column))
        for i in range(row):
            for j in range(column):
                distance[i][j] = self.cal_dis(lat_block[i], lon_block[j], target_location[0], target_location[1])
        max_value = np.amax(distance)
        for i in range(row):
            for j in range(column):
                distance[i][j] = distance[i][j] / max_value
        weight = np.zeros(num_feature)
        count = -1
        for i in range(row):
            for j in range(column):
                if position[i][j] == 0:
                    count = count + 1
                    weight[count] = distance[i][j]
        if count!=(num_feature-1):
            print('the weights and the number of features are not matched.')
            quit()

        X_w = np.zeros((Tr_sample, num_feature))
        for i in range(num_feature):
            for j in range(Tr_sample):
                X_w[j][i] = X_r[j][i] / weight[i]

        num_lambda=len(lambda_seq)
        mse=np.zeros(num_lambda)
        mse_train=np.zeros(num_lambda)
        for i in range(num_lambda):
            beta=self.fit_lasso_regression(X_w,Train[1],lambda_seq[i])
            beta_update=np.zeros(beta.shape)
            for j in range(num_feature):
                beta_update[j]=beta[j]/weight[j]
            y_pre=self.predict_lasso_regression(Validation[0],beta_update)
            y_train=self.predict_lasso_regression(X_w,beta)
            mse[i]=mean_squared_error(Validation[1],y_pre)
            mse_train[i] = mean_squared_error(Train[1], y_train)
        idx=np.argmin(mse)
#        print("validation:")
#        print(np.sqrt(mse[idx]))
#        print("train:")
#        print(np.sqrt(mse_train[idx]))
        return lambda_seq[idx]

    def fit_alasso_regression(self,X,Y,lambda_alasso,target_name):
        dict = {'Brazil': np.array([-10.0, 310.0]), 'Peru': np.array([-5.75, 283.0]), 'Asia': np.array([-10.0, 137.0])}
        target_location = dict[target_name]
        resolution = {332: [10, 'lat_lon_10x10.mat'], 1257: [5, 'lat_lon_5x5.mat'], 5881: [2.5, 'lat_lon_2.5x2.5.mat']}
        [Tr_sample, num_feature] = X.shape
        step = resolution[num_feature][0]
        data = loadmat(resolution[num_feature][1])
        position = data['lat_lon_data']
        [row, column] = position.shape
        lat_block = np.zeros(row)
        lon_block = np.zeros(column)
        for i in range(row):
            lat_block[i] = -90 + step * i
        for i in range(column):
            lon_block[i] = 0 + i * step
        distance = np.zeros((row, column))
        for i in range(row):
            for j in range(column):
                distance[i][j] = self.cal_dis(lat_block[i], lon_block[j], target_location[0], target_location[1])
        max_value = np.amax(distance)
        for i in range(row):
            for j in range(column):
                distance[i][j] = distance[i][j] / max_value
        weight = np.zeros(num_feature)
        count = -1
        for i in range(row):
            for j in range(column):
                if position[i][j] == 0:
                    count = count + 1
                    weight[count] = distance[i][j]
        if count != (num_feature-1):
            print('the weights and the number of features are not matched.')
            quit()
        X_w = np.zeros((Tr_sample, num_feature))
        for i in range(num_feature):
            for j in range(Tr_sample):
                X_w[j][i] = X[j][i] / weight[i]
        beta=self.fit_lasso_regression(X_w,Y,lasso_lambda=lambda_alasso)
        beta_update = np.zeros(beta.shape)
        for j in range(num_feature):
            beta_update[j] = beta[j] / weight[j]
        return beta_update

    def predict_lasso_regression(self,X,beta):
        return np.matmul(X,beta)

    def fit_lasso_regression(self, X, Y, lasso_lambda):
        clf = Lasso(alpha=lasso_lambda, fit_intercept=False, normalize=False, max_iter=10000, tol=1e-5)
        clf.fit(X, Y)
        return clf.coef_

    def predict_alasso_regression(self,X,beta):
        return np.matmul(X,beta)

    def run_alasso(self):
        X=pickle.load(open(os.path.join(self.dir,self.x_data_file),'rb'))
        Y=pickle.load(open(os.path.join(self.dir,self.y_data_file),'rb'))

        Y_pre_train_alasso = []
        Y_pre_test_alasso = []
        target_nums=len(self.target)
        beta_record=[]
        for climate_model in range(self.model_nums):
            print(climate_model)
            num_window = len(X[climate_model])
            target_train_alasso = []
            target_test_alasso = []
            target_beta=[]
            for target_idx in range(target_nums):
                print(target_idx)
                temp_train_alasso = []
                temp_test_alasso = []
                temp_beta=[]
                target_name=self.target[target_idx]
                for window in range(num_window):
                    print(window)
                    X_train = copy.deepcopy(X[climate_model][window][0])
                    X_test = copy.deepcopy(X[climate_model][window][1])
                    Y_train = copy.deepcopy(Y[climate_model][window][0][:, target_idx])
                    Y_test = copy.deepcopy(Y[climate_model][window][1][:, target_idx])
                    best_lambda=self.alasso_validation(X_train,Y_train,self.lambda_seq,self.validation_split,target_name)
                    print(best_lambda)
                    beta=self.fit_alasso_regression(X_train,Y_train,best_lambda,target_name)
                    temp_beta.append(beta)
                    pre_train=self.predict_alasso_regression(X_train,beta)
                    pre_test=self.predict_alasso_regression(X_test,beta)
                    temp_train_alasso.append(pre_train)
                    temp_test_alasso.append(pre_test)
                target_train_alasso.append(temp_train_alasso)
                target_test_alasso.append(temp_test_alasso)
                target_beta.append(temp_beta)
            Y_pre_train_alasso.append(target_train_alasso)
            Y_pre_test_alasso.append(target_test_alasso)
            beta_record.append(target_beta)
            result=[Y_pre_train_alasso,Y_pre_test_alasso]

        file=os.path.join(self.result_path, self.result_file)

        with open(file, 'wb') as f:
                pickle.dump(result, f)
#        return beta_record


class linear_alasso_corr(object):
    def __init__(self):
        self.lambda_seq=np.linspace(properties.ALasso_lambda_start,properties.ALasso_lambda_end,properties.ALasso_num_lambda)
        self.lambda_lasso=properties.ALasso_lambda
        self.x_data_file=properties.ALasso_X_file
        self.y_data_file=properties.ALasso_Y_file
        self.dir=properties.ALasso_Data_path
        self.validation_split=properties.ALasso_split
        self.model_nums = properties.ALasso_model_num
        self.target=properties.ALasso_target
        self.result_file=properties.ALasso_result_file_corr
        self.result_path=properties.ALasso_Result_path
    
    def alasso_validation(self,X,Y,lambda_seq,validation_split,target_name):
        [Train,Validation]=utility.train_validation(X,Y,validation_percentage=validation_split)
        X_r=Train[0]
        [Tr_sample,num_feature]=X_r.shape
        corr=np.zeros(num_feature)
        for i in range(num_feature):
            temp,p=pearsonr(X_r[:,i],Train[1])
            if(np.isnan(temp)):
                print(i)
                print("weight is nan")
            corr[i]=1/(1+abs(temp))
        max_value = np.amax(corr)
        weight=corr/max_value
        X_w = np.zeros((Tr_sample, num_feature))
        for i in range(num_feature):
            for j in range(Tr_sample):
                X_w[j][i] = X_r[j][i] / weight[i]
    
        num_lambda=len(lambda_seq)
        mse=np.zeros(num_lambda)
        for i in range(num_lambda):
            beta=self.fit_lasso_regression(X_w,Train[1],lambda_seq[i])
            beta_update=np.zeros(beta.shape)
            for j in range(num_feature):
                beta_update[j]=beta[j]/weight[j]
            y_pre=self.predict_lasso_regression(Validation[0],beta_update)
            mse[i]=mean_squared_error(Validation[1],y_pre)
        idx=np.argmin(mse)
        return lambda_seq[idx]
    
    def fit_alasso_regression(self,X,Y,lambda_alasso,target_name):
        [Tr_sample, num_feature] = X.shape
        corr=np.zeros(num_feature)
        for i in range(num_feature):
            temp,p=pearsonr(X[:,i],Y)
            if(np.isnan(temp)):
                print(i)
                print("weight is nan")
            corr[i]=1/(1+abs(temp))
        max_value = np.amax(corr)
        weight=corr/max_value
        X_w = np.zeros((Tr_sample, num_feature))
        for i in range(num_feature):
            for j in range(Tr_sample):
                X_w[j][i] = X[j][i] / weight[i]
        beta=self.fit_lasso_regression(X_w,Y,lasso_lambda=lambda_alasso)
        beta_update = np.zeros(beta.shape)
        for j in range(num_feature):
            beta_update[j] = beta[j] / weight[j]
        return beta_update
    
    def predict_lasso_regression(self,X,beta):
        return np.matmul(X,beta)
    
    def fit_lasso_regression(self, X, Y, lasso_lambda):
        clf = Lasso(alpha=lasso_lambda, fit_intercept=False, normalize=False, max_iter=10000, tol=1e-5)
        clf.fit(X, Y)
        return clf.coef_
    
    def predict_alasso_regression(self,X,beta):
        return np.matmul(X,beta)
    
    def run_alasso_corr(self):
        X=pickle.load(open(os.path.join(self.dir,self.x_data_file),'rb'))
        Y=pickle.load(open(os.path.join(self.dir,self.y_data_file),'rb'))
        
        Y_pre_train_alasso = []
        Y_pre_test_alasso = []
        target_nums=len(self.target)
        beta_record=[]
        for climate_model in range(self.model_nums):
            print(climate_model)
            num_window = len(X[climate_model])
            target_train_alasso = []
            target_test_alasso = []
            target_beta=[]
            for target_idx in range(target_nums):
                print(target_idx)
                temp_train_alasso = []
                temp_test_alasso = []
                temp_beta=[]
                target_name=self.target[target_idx]
                for window in range(num_window):
                    print(window)
                    X_train = copy.deepcopy(X[climate_model][window][0])
                    X_test = copy.deepcopy(X[climate_model][window][1])
                    Y_train = copy.deepcopy(Y[climate_model][window][0][:, target_idx])
                    Y_test = copy.deepcopy(Y[climate_model][window][1][:, target_idx])
                    best_lambda=self.alasso_validation(X_train,Y_train,self.lambda_seq,self.validation_split,target_name)
                    beta=self.fit_alasso_regression(X_train,Y_train,best_lambda,target_name)
                    temp_beta.append(beta)
                    pre_train=self.predict_alasso_regression(X_train,beta)
                    pre_test=self.predict_alasso_regression(X_test,beta)
                    temp_train_alasso.append(pre_train)
                    temp_test_alasso.append(pre_test)
                target_train_alasso.append(temp_train_alasso)
                target_test_alasso.append(temp_test_alasso)
                target_beta.append(temp_beta)
            Y_pre_train_alasso.append(target_train_alasso)
            Y_pre_test_alasso.append(target_test_alasso)
            beta_record.append(target_beta)
            result=[Y_pre_train_alasso,Y_pre_test_alasso]
        
        file=os.path.join(self.result_path, self.result_file)
        
        with open(file, 'wb') as f:
            pickle.dump(result, f)



