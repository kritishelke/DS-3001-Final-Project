''' 
Next Day Direction Model
Model type: Gradient-boosted trees

''' 

import pandas as pd
import numpy as np 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


#Balance fit vs. overfitting
n_estimators = 100
max_depth = 6
learning_rate = 0.1

#Add randomness to calm variance
subsample = 0.8
colsample_bytree = 0.8

#regularize noisy finance features
min_child_weight = 1
reg_alpha = 0
reg_lambda = 1