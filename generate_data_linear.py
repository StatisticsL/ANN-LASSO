'''
generate the data for linear model
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# plt.interactive(False)
import pandas as pd
import os
#############################################################################
###--------------------------parameters-----------------------------------
truemodel='linear'
dir_name=''+truemodel+'/'  ###add your directory
num_repli=100
SNR=3
fp=10
p1 = 200
n_train = 70
n_val=30
n_test=fp*(n_train+n_val)

###--------------------------generate data-----------------------------------
for s in range(17):
    beta=SNR*np.ones(s)
    Inds=np.zeros(shape=(num_repli,s))  
    for simu in range(num_repli):
        inds =np.random.choice(range(p1), s, replace=False)
        Inds[simu,:]=inds
        x_train=np.random.normal(size = (n_train, p1))
        y_train=np.dot(x_train[:,inds],beta)+np.random.normal(size=(n_train))
        
        x_val=np.random.normal(size = (n_val, p1))
        y_val=np.dot(x_val[:,inds],beta)+np.random.normal(size=(n_val))
        
        x_test=np.random.normal(size = (n_test, p1))
        y_test=np.dot(x_test[:,inds],beta)
        
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
    Inds=pd.DataFrame(Inds)###Inds  is used in python, it plus one = true index
    Inds.to_excel(dir_name+'s'+str(s)+'/'+'Inds.xlsx',header=False,index=False)
