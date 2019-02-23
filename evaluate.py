import pickle
from lcp_utility import utility
import numpy as np
from scipy import stats
# Change the path to where you save the results and groud truth
# Change following parameters accordingly
result=pickle.load(open("result_gbt5_2x2.p",'rb'))# estimated results
true=pickle.load(open("Y_sep5.p",'rb'))# ground truth
r2=np.zeros((3,3,2,10))# (#models,#regions,#train+test,#windows)
rmse=np.zeros((3,3,2,10))# (#models,#regions,#train+test,#windows )
sum_rmse=np.zeros((2,2,3,3))# (#train+test,#mean+se,#models,#regions)
sum_r2=np.zeros((2,2,3,3))
num_models=3
num_regions=3
# Evaluation
for i in range(2):
    for climate_model in range(num_models):
        for target_region in range(num_regions):
            num_window=len(true[climate_model])
            for window in range(num_window):
                print(window)
                a=result[i][climate_model][target_region][window]
                b=true[climate_model][int(window)][i][:, target_region]
                r2[climate_model][target_region][i][window]=utility.compute_r2(true[climate_model][int(window)][i][:,target_region],result[i][climate_model][target_region][window])
                rmse[climate_model][target_region][i][window] = utility.compute_rmse(true[climate_model][int(window)][i][:, target_region],result[i][climate_model][target_region][window])
            sum_rmse[i][0][climate_model][target_region]=np.average(rmse[climate_model][target_region][i][0:num_window],axis=0)
            sum_rmse[i][1][climate_model][target_region] = stats.sem(rmse[climate_model][target_region][i][0:num_window], axis=0)
            sum_r2[i][0][climate_model][target_region]=np.average(r2[climate_model][target_region][i][0:num_window],axis=0)
            sum_r2[i][1][climate_model][target_region] = stats.sem(r2[climate_model][target_region][i][0:num_window], axis=0)
#save evaluation results
np.save("rmse_gbt_new.npy",sum_rmse)
np.save("r2_gbt_new.npy",sum_rmse)

