# Random_Forest_for_MHW
Using Random Forest to predict Marine Heatwaves in the North Pacific

Usage: 

1. Regridding.m : Matlab script used to lower the resolution of OISST from 1/4deg to 2.5x2.5 degrees to match the NCEP-NCAR predictors used.
2. mav_lag_bal.m :  Matlab script to calculate: (1) moving average using 7 days before the day of interest for predictors; (2) multiple lags between predictors and MHW occurences; (3) balanced occurences of the 4 categories of MHW severity. 
3. RandomForest_1902.ipynb : IPython notebook used to make the train-test datasets, construct the random forest model, run randomised cross validation and plot condusion matrix, feature importance, ROC and Precision-Recall curves. 
4. Prediction_Maps: Folder including all necessary data and IPython notebook to create the AUC vs time lags plot and the maps that represent an example of predicted and observed MHWs in northeast Pacific. 
5. Final_Model_Figure : Folder containing all data and IPython notebook to plot the paper's figures. 
