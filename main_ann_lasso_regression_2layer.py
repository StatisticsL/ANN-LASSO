'''
ann_lasso for regression

'''
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
import scipy.stats as ss
import sklearn.preprocessing #pip install sklearn
from sklearn import linear_model
import pandas as pd
import numpy as np
from sys import argv
import matplotlib as plt
import json
import csv
from ann_lasso_regression_2layer import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#############################################################################
method_name='ann_lasso'
truemodel='nonlinear'
num_repli=1
dir_name_results=''###add your directory of your results
dir_name=''###add your directory of your datasets
s=4
p1=50
p2=20
Inds=pd.read_excel(dir_name+truemodel+'/s'+str(s)+'/Inds.xlsx',header=None)
Inds=Inds.values.astype(float)
Index=np.zeros(shape=(num_repli,p1))
Fdr=np.zeros(shape=(num_repli,1))
Tpr=np.zeros(shape=(num_repli,1))
Pe=np.zeros(shape=(num_repli,1))
for simu in range(num_repli):
    x_train=pd.read_excel(dir_name+truemodel+'/s'+str(s)+'/'+'x_train_'+str(simu)+'.xlsx',header=None)
    x_val=pd.read_excel(dir_name+truemodel+'/s'+str(s)+'/'+'x_val_'+str(simu)+'.xlsx',header=None)
    x_test=pd.read_excel(dir_name+truemodel+'/s'+str(s)+'/'+'x_test_'+str(simu)+'.xlsx',header=None)
    y_train=pd.read_excel(dir_name+truemodel+'/s'+str(s)+'/'+'y_train_'+str(simu)+'.xlsx',header=None)
    y_val=pd.read_excel(dir_name+truemodel+'/s'+str(s)+'/'+'y_val_'+str(simu)+'.xlsx',header=None)
    y_test=pd.read_excel(dir_name+truemodel+'/s'+str(s)+'/'+'y_test_'+str(simu)+'.xlsx',header=None)
    
    x_train=pd.concat([x_train,x_val],axis=0)
    y_train=pd.concat([y_train,y_val],axis=0)
    del x_val,y_val
    
    x_train=x_train.values.astype(float)
    x_test=x_test.values.astype(float) 
    y_train=y_train.values.astype(float)
    y_test=y_test.values.astype(float)
    y_train=y_train.T
    y_test=y_test.T
    
    
    
    n_train=x_train.shape[0]
    x_train_l2norm=tf.sqrt(tf.reduce_sum(tf.cast(x_train**2,tf.float64),axis=0))
    x_train_l2norm=x_train_l2norm.numpy()
    x_train_rescaled=x_train/np.repeat(x_train_l2norm.reshape(1,p1),n_train,axis=0)
    x_train_rescaled=np.concatenate((x_train_rescaled,np.ones([n_train,1])), axis=1)
    n_test=x_test.shape[0]
    x_test_rescaled=x_test/np.repeat(x_train_l2norm.reshape(1,p1),n_test,axis=0)
    x_test_rescaled=np.concatenate((x_test_rescaled,np.ones([n_test,1])), axis=1)
    
    lambdaqut_SANN = lambda_qut_sann_regression(tf.nn.l2_normalize(x_train,axis=0),nSample=10000,miniBatchSize=500,alpha=0.05,option='quantile')
      
    ipath=-1
    lambi=np.exp(ipath)/(1+np.exp(ipath))*lambdaqut_SANN
    [w1o,w2o,b2o,yhato,mutesthato,costo,lro_o]=TFmodelfit(x_train_rescaled,y_train,p2,activation_function,lambi,[0.1, 0.01,0.001,0.0001,0.00001],True,ifTrain=True,iterationISTA=3000, Xtest=None, iniscale=0.01)
    for ipath in range(5):
        lambi=np.exp(ipath)/(1+np.exp(ipath))*lambdaqut_SANN
        [w1o,w2o,b2o,yhato,mutesthato,costo,lro_o]=TFmodel(x_train_rescaled,y_train,p2,activation_function,lambi,lro_o,True,ifTrain=True,iterationISTA=3000, Xtest=None,
            rel_err=1.e-9, w1_ini=w1o, w2_ini=w2o, b2_ini=b2o)
    [w1o,w2o,b2o,yhato,mutesthato,costo,lro_o]=TFmodel(x_train_rescaled,y_train,p2,activation_function,lambdaqut_SANN,lro_o,False,ifTrain=True,iterationISTA=10000, Xtest=x_test_rescaled,
        rel_err=1.e-12, w1_ini=w1o, w2_ini=w2o, b2_ini=b2o)
    
    if(s==0):
        needles_index_hat=np.array(np.where(np.sum(abs(w1o[:,0:(p1-1)]),axis=0)>0),dtype=float)+1
        fdr=len(needles_index_hat[0])/np.max((len(needles_index_hat[0]),1)) 
        tpr=1
        ###########################y_test shape!!!
        pe=tf.reduce_sum(tf.square(y_test-mutesthato),axis=1)/y_test.shape[1]
    else:
        needles_index_hat,tpr,fdr,hat_equal_true,pe=resultanalysis(w1o,y_test,mutesthato,needles_index_true=Inds[simu,:]+1)      
    
    Index[simu,0:len(needles_index_hat[0])]=needles_index_hat[0]
    Fdr[simu,0]=fdr
    Tpr[simu,0]=tpr
    Pe[simu,0]=pe
Index[np.where(Index==0)]=None 
Index=pd.DataFrame(Index)
Fdr=pd.DataFrame(Fdr)
Tpr=pd.DataFrame(Tpr)
Pe=pd.DataFrame(Pe)

Index.to_excel(dir_name_results+truemodel+'/'+method_name+'_Index_'+str(s)+'.xlsx',header=None,index=None)        
Fdr.to_excel(dir_name_results+truemodel+'/'+method_name+'_Fdr_'+str(s)+'.xlsx',header=None,index=None) 
Tpr.to_excel(dir_name_results+truemodel+'/'+method_name+'_Tpr_'+str(s)+'.xlsx',header=None,index=None) 
Pe.to_excel(dir_name_results+truemodel+'/'+method_name+'_Pe_'+str(s)+'.xlsx',header=None,index=None)       
