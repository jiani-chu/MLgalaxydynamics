import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator
from optparse import OptionParser
from scipy.integrate import quad
# from sympy import *
import pickle
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pandas as pd
import seaborn as sns
import lgb_data_New
labelkey='mtot'
labelKEY=r'$\log_{10}\,M_{\rm tot}\,$[$\mathrm{M_{\odot}}$]'
# labelkey='mdm'
# labelKEY=r'$\log_{10}\,M_{\rm DM}\,$[$\mathrm{M_{\odot}}$]'
# labelkey='mstar'
# labelKEY=r'$\log_{10}\,M_{*}\,$[$\mathrm{M_{\odot}}$]'
# labelKEY_nano=r'$M_{*}$'
path='./'

mag=r'$M_{ r}$'
vmean=r'$\bar{v}$'#r'$log_{10}[\bar{v}\rm km/s]$'
vsigma=r'$\sigma_v$'#r'$log_{10}[\sigma_v\rm km/s]$'
Lambda=r'$\lambda$'
SFR=r'$SFR$'#r'$SFR \rm M_{\odot}/h/yr$'
Ser_index=r'$n_{\rm Ser}$'
Fcold=r'$f_{cold}$'
Fhot=r'$f_{hot}$'
B2T=r'$B/T$'
color='g-r'
C2a=r'$c/a$'

params = {
    'num_leaves': 5,
    'objective': 'regression',
    'metric': ['l1', 'l2'],
    'verbose': -1
}


## all,with Mag
X,Y=lgb_data_New.lgb_data(labelkey,'New')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
y_train=np.reshape(y_train,-1)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)



evals_result = {}  # to record eval results for plotting
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                #                 valid_sets=[lgb_train, lgb_test],
                feature_name=[f'f{i + 1}' for i in range(X_train.shape[-1])],
                callbacks=[
                    lgb.record_evaluation(evals_result)
                ])


y_pred = gbm.predict(X_test)

Nte=len(y_pred)
var_ratio=y_pred.reshape(-1)-y_test.reshape(-1)
q75, q25 = np.percentile(var_ratio, [75 ,25])
var_delta=2*(q75-q25)/(Nte**(1/3))

importance = gbm.feature_importance(importance_type='gain')
feature_name = gbm.feature_name()
feature_name = [mag, vmean, vsigma, Lambda, SFR, Ser_index, B2T, Fcold, Fhot, color, C2a]
plot_importance = [importance[np.argsort(-importance)[0]],
                   importance[np.argsort(-importance)[1]],
                   importance[np.argsort(-importance)[2]],


                   np.sum(importance[np.argsort(-importance)[3:]])]
plot_feature_name = [feature_name[np.argsort(-importance)[0]],
                     feature_name[np.argsort(-importance)[1]],
                     feature_name[np.argsort(-importance)[2]],



                     'other']

data={'y_test':y_test,'y_pred':y_pred,'var_ratio':var_ratio,'var_delta':var_delta,'plot_feature_name':plot_feature_name,'plot_importance':plot_importance}
with open(path+labelkey+'_withmag.dat', 'wb') as f:
    pickle.dump(data, f)


## all,with Mag
X,Y=lgb_data_New.lgb_data(labelkey,'New_noMag')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
y_train=np.reshape(y_train,-1)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)



evals_result = {}  # to record eval results for plotting
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                #                 valid_sets=[lgb_train, lgb_test],
                feature_name=[f'f{i + 1}' for i in range(X_train.shape[-1])],
                callbacks=[
                    lgb.record_evaluation(evals_result)
                ])


y_pred = gbm.predict(X_test)

Nte=len(y_pred)
var_ratio=y_pred.reshape(-1)-y_test.reshape(-1)
q75, q25 = np.percentile(var_ratio, [75 ,25])
var_delta=2*(q75-q25)/(Nte**(1/3))

importance = gbm.feature_importance(importance_type='gain')
feature_name = gbm.feature_name()
feature_name = [mag, vmean, vsigma, Lambda, SFR, Ser_index, B2T, Fcold, Fhot, color, C2a]
plot_importance = [importance[np.argsort(-importance)[0]],
                   importance[np.argsort(-importance)[1]],
                   importance[np.argsort(-importance)[2]],



                   np.sum(importance[np.argsort(-importance)[3:]])]
plot_feature_name = [feature_name[np.argsort(-importance)[0]],
                     feature_name[np.argsort(-importance)[1]],
                     feature_name[np.argsort(-importance)[2]],



                     'other']

data={'y_test':y_test,'y_pred':y_pred,'var_ratio':var_ratio,'var_delta':var_delta,'plot_feature_name':plot_feature_name,'plot_importance':plot_importance}
with open(path+labelkey+'_withoutmag.dat', 'wb') as f:
    pickle.dump(data, f)



