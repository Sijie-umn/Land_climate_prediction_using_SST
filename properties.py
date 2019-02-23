# -*- coding: utf-8 -*-
# Parameters for preprocessing
Data_path=r'/Users/sijiehe/Research/Summer 2018/climate_data'#Path of raw data
Train_size=100 #years of data for each training set
Test_size=10 #years of data for each test set
Step_size=5 #the stride of sliding window

# Parameters for linear model - Principal component regression

PCR_X_file='X_sep5_2x2.p'# preprocessed sea surface temperature data file
PCR_Y_file='Y_sep5.p'# preprocessed land temperatue data file
PCR_Data_path=r'/Users/sijiehe/Research/Summer 2018/climate_data'#Path of the above two files
PCR_split=0.2 #the percentage for validation set (e.g. use the last 20% data as validation set for hyper-parameter tuning)
PCR_model_num=3 #number of climate models
PCR_target_num=3 #number of land target regions
PCR_result_file="result_PCR5_2x2.p" #predicted land temperature file name
PCR_Result_path=r'/Users/sijiehe/Research/Summer 2018/test' #path for saving results
PCR_thre_var=0.9#The threshold of PCR (Choose the
PCR_step_size=2 #step size for finding the best number of PCs




# Parameters for linear model - Lasso
Lasso_X_file='X_sep5_2x2.p'# preprocessed sea surface temperature data file
Lasso_Y_file='Y_sep5.p'# preprocessed land temperatue data file
Lasso_Data_path=r'/Users/sijiehe/Research/Summer 2018/climate_data'#Path of the above two files
Lasso_Result_path=r'/Users/sijiehe/Research/Summer 2018/test'#path for saving results
Lasso_lambda_start=0.01 # For hyperparamete tuning, the smallest lambda
Lasso_lambda_end=0.5 # For hyperparamete tuning, the largest lambda
Lasso_num_lambda=40 #For hyperparamete tuning, the number of lambdas
Lasso_lambda=0.1
Lasso_split=0.2#the percentage for validation set
Lasso_model_num=3 #number of climate models
Lasso_target_num=3 #number of land target regions
Lasso_result_file="result_lasso5_2x2.p"#predicted land temperature file name




# Parameters for linear model - Adaptive Lasso

ALasso_X_file='X_sep5_2x2.p'# preprocessed sea surface temperatue data file
ALasso_Y_file='Y_sep5.p' # preprocessed land temperatue data file
ALasso_Data_path=r'/Users/sijiehe/Research/Summer 2018/climate_data'#Path of the above two files
ALasso_Result_path=r'/Users/sijiehe/Research/Summer 2018/test' #path for saving results
ALasso_lambda_start=0.01 # For hyperparamete tuning, the smallest lambda
ALasso_lambda_end=0.5 # For hyperparamete tuning, the largest lambda
ALasso_num_lambda=40 #For hyperparamete tuning, the number of lambdas
ALasso_lambda=0.1
ALasso_split=0.2#the percentage for validation set
ALasso_model_num=3 #number of models
#ALasso_target=['Brazil']
ALasso_target=['Brazil','Peru','Asia'] #Land target regions
ALasso_result_file="result_alasso5_2x2.p" #predicted land temperature file name
ALasso_result_file_corr="result_alasso_corr5_2x2.p" #predicted land temperature file name (using corrleation as weights)


#Parameters for non linear model
DATA_PATH =r'/Users/sijiehe/Research/Summer 2018/climate_data'#path for data files
#'../../../../project/banerjee-00/datasets/SDM_2012_climate_model_data/Experiment_datasets/climate_data'
FILE_X =  'X_sep5_2x2.p'#preprocessed sea surface temperatue data file
FILE_Y = 'Y_sep5.p'# preprocessed land temperatue data file
RES_PATH=r'/Users/sijiehe/Research/Summer 2018/test'#path for saving results


RES_FILE_MLP = 'result_mlp5_2x2.p' #file name of results using deep nets
RES_FILE_GBT = 'result_gbt5_2x2.p' #file name of results using gradient bootsted tree

NO_MODELS = 3 #number of models
NO_LOC = 3 #number of land target regions

# Parameters for non linear model - MLP

NO_EPOCH = 100 #number of epoch for training

NO_LAYERS = 4 #number of hidden layers
NO_HIDDEN_UNITS = 32 #number of hidden units
BATCH_SIZE = 32 #number of batch size


OPTIMIZER = 'Adam' #optimizer
REGULARIZER= 'L1' #regularization: 'L1' , 'L2', 'L1_l2' or None


EARLY_STOP = True # Early stop
LEARNING_RATE_DECAY = False # learninig rate decay


# Parameters for gradient boosted tree(xgboost)- GBT
NO_ROUND = 100 #number of trees
ETA = 2 #
MAX_DEPTH = 4 #maximum depth of each tree

EVAL_METRIC = 'rmse' #evaluation metric while training

# Parameters for non linear model - Convolutional neural networks
# Parameters for non linear model - transfer learning

