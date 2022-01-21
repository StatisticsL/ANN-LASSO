'''
generate the data
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#############################################################################
###--------------------------parameters-----------------------------------
truemodel='nonlinear'
num_repli=100
dir_name=''+truemodel+'/'###add your directory
SNR=10
fp=10
p1 = 50
n_train=350
n_val=150
n_test=fp*(n_train+n_val)

###---------------------------generate data-----------------------------------
for h in range(9):
    s=h*2
    Inds=np.zeros(shape=(num_repli,s))
    print('s= '+str(s))
    
    for simu in range(num_repli):
        inds=np.random.choice(range(p1),s,replace=False)
        Inds[simu,:]=inds
        x_train=np.random.normal(size=(n_train,p1))
        x_val=np.random.normal(size=(n_val,p1))
        x_test=np.random.normal(size=(n_test,p1))
        y_train=np.zeros(n_train)
        y_val=np.zeros(n_val)
        y_test=np.zeros(n_test)
        
        for k in range(h):
            y_train=y_train+np.abs(x_train[:,inds[k]]-x_train[:,inds[k+h]])
            y_val=y_val+np.abs(x_val[:,inds[k]]-x_val[:,inds[k+h]])
            y_test=y_test+np.abs(x_test[:,inds[k]]-x_test[:,inds[k+h]])
        
        y_train=SNR*y_train+np.random.normal(size=n_train)
        y_val=SNR*y_val+np.random.normal(size=n_val)
        y_test=SNR*y_test
        
        x_train=pd.DataFrame(x_train)
        x_val=pd.DataFrame(x_val)
        x_test=pd.DataFrame(x_test)
        y_train=pd.DataFrame(y_train)
        y_val=pd.DataFrame(y_val)
        y_test=pd.DataFrame(y_test)
        
        x_train.to_excel(dir_name+'s'+str(s)+'/'+'x_train_'+str(simu)+'.xlsx',header=False,index=False)
        x_val.to_excel(dir_name+'s'+str(s)+'/'+'x_val_'+str(simu)+'.xlsx',header=False,index=False)
        x_test.to_excel(dir_name+'s'+str(s)+'/'+'x_test_'+str(simu)+'.xlsx',header=False,index=False)
        y_train.to_excel(dir_name+'s'+str(s)+'/'+'y_train_'+str(simu)+'.xlsx',header=False,index=False)
        y_val.to_excel(dir_name+'s'+str(s)+'/'+'y_val_'+str(simu)+'.xlsx',header=False,index=False)
        y_test.to_excel(dir_name+'s'+str(s)+'/'+'y_test_'+str(simu)+'.xlsx',header=False,index=False)
        del x_train,x_val,x_test,y_train,y_test,y_val
        print('               simu= '+str(simu))
    Inds=pd.DataFrame(Inds)###Inds  is used in python, it plus one = true index
    Inds.to_excel(dir_name+'s'+str(s)+'/'+'Inds.xlsx',header=False,index=False)