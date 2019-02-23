# Land climate predicition with sea surface temperature (SST)

## Introduction

See [the AAAI-19 paper](https://www.aaai.org/Papers/AAAI/2019/AAAI-HeSijie.6971.pdf) for technical information on the land climate prediction with sea surface temperature.


## Getting started

1. Installing the necessary python packages

   - required python version is python 3.6.x
   - required packages:
      - numpy==1.14.3
      - pickle=4.0
      - sklearn=0.19.1
      - scipy=1.1.0
      - keras=2.2.4
      - xgboost=0.81

2. Preparing data for the model

    [The original dataset](https://journals.ametsoc.org/doi/10.1175/BAMS-D-11-00094.1) is stored [here](https://drive.google.com/open?id=15Atnsi8VOy6QQ5fS_uPifxI9r5VFHVmo).
    Please refer to ~/Land_climate_prediction_SST/data/data_description.txt for the detailed descripiton. 
    Data preprocessing can be done by directly running 
    ```
    python3 ~/Land_climate_prediction_SST/lcp_preprocessing/preprocess_seperate.py
    ```
    Note please change the path to data files by revising ~/Land_climate_prediction_SST/properties.py.

3. Running the model

    1. Change the settings and hyper-parameters in ~/Land_climate_prediction_SST/properties.py. 
    2. Run the model by running ~/Land_climate_prediction_SST/main.py. It will return the predicted land temperature for both training set and test set.
        ```
        python3 main.py
        ```
    3. Evaluate the results by running ~/Land_climate_prediction_SST/evaluate.py. Note please change the path to the results accordingly.


  
  

