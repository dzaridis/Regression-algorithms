#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import Lasso
from sklearn.linear_model import MultiTaskLasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from numpy import absolute
from numpy import mean
from numpy import std
from glob import glob
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold,RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler 
import time
from sklearn.multioutput import MultiOutputRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor


# In[2]:


def ExtraTreeRegr(x,y):
    """
    Extra tree regression algorithm on preprocessed data (polynomial features, PCA and standard scaler used for data preprocessing)
    Comparison of grid search effectiveness on scores
    Args:
    x (np.array or series): the independant variables to train the model
    y (np.array or series): the dependent variables to predict
    Outpout:
    Prints the best parameters and scores for mean squared error and explained variance after applying exhaustive grid
    search and 3 repeated 10fold cross validation.
    """
    selector = VarianceThreshold()
    scaler = StandardScaler()
    pca = PCA()
    sgdr = ExtraTreesRegressor()
    pipeline = Pipeline(steps=[('selector', selector), ('scaler', scaler),  ('pca', pca), ('model', sgdr)],memory = 'tmp')
    pipeline_null = Pipeline(steps=[('model', sgdr)],memory = 'tmp')
    param_grid = {
        'pca__n_components': [2,5,8],
        'pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
        #'selector__threshold' : [0.5, 1, 2],
        "scaler__with_mean":[True],
        "scaler__with_std": [False],
        "model__n_estimators": [30,50,100,200]
    }
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    start_time=time.time()
    grid_mse = (GridSearchCV(pipeline,param_grid,scoring='neg_mean_squared_error', cv=5))
    grid_ev = (GridSearchCV(pipeline,param_grid,scoring='neg_root_mean_squared_error', cv=5))
    
    result_mse = grid_mse.fit(x, y)
    result_ev = grid_ev.fit(x, y)
    ens_hsgdr_hard_time=time.time() - start_time
    ens_hsgdr_regr_res_mse_i=result_mse.best_score_
    ens_hsgdr_regr_par_mse_i=result_mse.best_params_
    ens_hsgdr_regr_res_ev_i=result_ev.best_score_
    ens_hsgdr_regr_par_ev_i=result_ev.best_params_

    print("-----------------Grid search-----------------")
    print("Time: %s" % ens_hsgdr_hard_time)
    print('Best Score: %s' % absolute(result_mse.best_score_))
    print('Best Hyperparameters: %s' % result_mse.best_params_)
    print('Best Score: %s' % absolute(result_ev.best_score_))
    print('Best Hyperparameters: %s' % result_ev.best_params_)
    
    start_time=time.time()
    grid_mse = (GridSearchCV(pipeline_null,param_grid_null,scoring='neg_mean_squared_error', cv=5))
    grid_ev = (GridSearchCV(pipeline_null,param_grid_null,scoring='neg_root_mean_squared_error', cv=5))

    result_mse = grid_mse.fit(x, y)
    result_ev = grid_ev.fit(x, y)
    ens_hsgdr_hard_time_null=time.time() - start_time
    ens_hsgdr_regr_res_mse_i_null=result_mse.best_score_
    ens_hsgdr_regr_par_mse_i_null=result_mse.best_params_
    ens_hsgdr_regr_res_ev_i_null=result_ev.best_score_
    ens_hsgdr_regr_par_ev_i_null=result_ev.best_params_

    print("-----------------Null grid-----------------")
    print("Time: %s" % ens_hsgdr_hard_time_null)
    print('Best Score: %s' % absolute(result_mse.best_score_))
    print('Best Hyperparameters: %s' % result_mse.best_params_)
    print('Best Score: %s' % absolute(result_ev.best_score_))
    print('Best Hyperparameters: %s' % result_ev.best_params_)


# In[ ]:




