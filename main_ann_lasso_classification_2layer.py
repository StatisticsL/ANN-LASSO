"""
ann_lasso_classification_2layer
"""
import tensorflow as tf
import numpy as np 
import time 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from ann_lasso_classification_2layer import *
import pywt####pip3 install PyWavelets
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.utils import resample
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
######################################################################### 
time_start=time.time()


dir_name_data=''  ###add your directory of your data sets
data_name_all=os.listdir(dir_name_data)
data_name='alon'
dir_name_results='' ###add your directory of your results
method_name='ann_lasso_'

###########################################################################
x_train=pd.read_csv(dir_name_data+data_name+'/x_train.csv',header=None)
x_test=pd.read_csv(dir_name_data+data_name+'/x_test.csv',header=None)
y_train=pd.read_csv(dir_name_data+data_name+'/y_train.csv',header=None)
y_test=pd.read_csv(dir_name_data+data_name+'/y_test.csv',header=None)

x_train=x_train.values.astype(float)
x_test=x_test.values.astype(float)
n_train=x_train.shape[0]

y= pd.concat([y_train,y_test],axis=0)
encoder = LabelEncoder()
encoded_y = encoder.fit_transform(y)
dummy_y = np_utils.to_categorical(encoded_y)

y_train=dummy_y[0:n_train,:]
y_test=dummy_y[n_train:,]
y_train=np.transpose(y_train)
y_test=np.transpose(y_test)
y_train=y_train.astype('float64')
y_test=y_test.astype('float64')

del n_train,y,encoder,encoded_y,dummy_y
print('Completed loading data')
#########################################################################   
filename_confusion=dir_name_results+'confusion_matrix/'+method_name+'_'+data_name+'_confusion_matrix.txt'
filename_accuracy=dir_name_results+'accuracy/'+method_name+'_'+data_name+'_accuracy.txt'
filename_needles=dir_name_results+'needles_index/'+method_name+'_'+data_name+'_needles_index.txt'
filename_num_needles=dir_name_results+'num_needles/'+method_name+'_'+data_name+'_num_needles.txt'


n_train,p1=x_train.shape
x_train_l2norm=tf.sqrt(tf.reduce_sum(tf.cast(x_train**2,tf.float64),axis=0))
x_train_l2norm=x_train_l2norm.numpy()
x_train_rescaled=x_train/np.repeat(x_train_l2norm.reshape(1,p1),n_train,axis=0)
x_train_rescaled=np.concatenate((x_train_rescaled,np.ones([n_train,1])), axis=1)
n_test=x_test.shape[0]
x_test_rescaled=x_test/np.repeat(x_train_l2norm.reshape(1,p1),n_test,axis=0)
x_test_rescaled=np.concatenate((x_test_rescaled,np.ones([n_test,1])), axis=1)


hat_p_training = tf.reduce_mean(tf.cast(y_train,tf.float64),axis=1)
nSample=10000
lambda_qut=lambda_qut_sann_classification(tf.nn.l2_normalize(x_train,axis=0),hat_p_training,nSample=nSample,miniBatchSize=50,alpha=0.05,option='quantile')
p2=20
num_rep = 1
learningRate_list=[0.1,0.01,0.001,0.0001,0.00001]
iniscale=0.01


for repli in np.arange(num_rep):
    w1o,w2o,co,p_hat_o,y_hat_o,p_test_hat_o,y_test_hat_o,cost_o,lro,needles_index_hat,tpr,fdr,hat_equal_true,confusion_matrix,accuracy=computeResults(x_train_rescaled,y_train,learningRate_list,iniscale,lambda_qut,p2,x_test_rescaled,y_test) 
    with open(filename_confusion,'a') as file_con:
        file_con.write('the '+str(repli)+' time'+"\n")
        file_con.write(str(confusion_matrix))
        file_con.write("\n")
    with open(filename_accuracy,'a') as file_acc:
        file_acc.write(str(accuracy))
        file_acc.write("\n")
    with open(filename_needles,'a') as file_needlesindex:
        file_needlesindex.write(str(needles_index_hat))
        file_needlesindex.write("\n")
    with open(filename_num_needles,'a') as file_numneedles:
        file_numneedles.write(str(needles_index_hat.shape[1]))
        file_numneedles.write("\n")   
    filename=dir_name_results+"SANN/"+data_name+"_result_lambdaqut_"+str(nSample)+"_p2_"+str(p2)+"_"+str(repli)+"_cost_"+str(cost_o.numpy())+".txt"
    writeResults(filename,p2,nSample,lambda_qut,cost_o,confusion_matrix,y_test,accuracy,needles_index_hat,w1o,w2o,co)        
    del w1o,w2o,co,p_hat_o,y_hat_o,p_test_hat_o,y_test_hat_o,cost_o,lro,accuracy
    del needles_index_hat,tpr,fdr,hat_equal_true,confusion_matrix,filename  
    
del filename_confusion,filename_accuracy,filename_needles,filename_num_needles
del hat_p_training,nSample,lambda_qut,p2,learningRate_list,iniscale
del x_train,x_test,y_train,y_test
del file_acc,file_con,file_needlesindex,file_numneedles
####################################################################################
time_end=time.time()
print("It takes " + str(time_end-time_start) + " seconds for "+str(num_rep)+" times." )

