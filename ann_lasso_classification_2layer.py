"""
simulations for classification via ann_lasso
"""
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###############################################################################
def data_form_processing(x_data,y_data,test_ratio,del_zero_colume=False,
      x_isDataFrame=True,y_isDataFrame=True):
    """
    x_data : n*p1,no tile,dataframe
    y_data : n*1,no tile,dataframe
        DESCRIPTION.
    test_ratio : 0~1
    """
    if(x_isDataFrame==False):
        x_data=pd.DataFrame(x_data)
    if(y_isDataFrame==False):
        y_data=pd.DataFrame(y_data) 
    y_data.rename(columns={0:'class'},inplace=True)
    labels=np.unique(y_data.values)
    number_class=len(labels)
    ####删除含有零元的列
    if(del_zero_colume==True):
        del_index_colume=np.unique(np.where(x_data.values==0)[1])
        del_head_colume=x_data.columns.values[del_index_colume]
        x_data=x_data.drop(columns=del_head_colume)
        
    data = pd.concat([y_data,x_data],axis=1)
    p1=data.values.shape[1]
    
    #####随机排序
    n_total=data.values.shape[0]
    data=data.take(np.random.permutation(n_total))

    ###分层抽样
    X_train, X_test, Y_train, Y_test = train_test_split(data,data['class'],test_size=test_ratio,random_state=1010,stratify=data['class'])

    x_train=X_train.values[:,1:p1]
    x_test=X_test.values[:,1:p1]
    x_train=x_train.astype('float64')
    x_test=x_test.astype('float64')
    
    Y_train=Y_train.values
    Y_test=Y_test.values
    n_train=len(Y_train)
    n_test=len(Y_test)
    y_train = np.zeros([number_class,n_train])
    y_test = np.zeros([number_class,n_test])
    for i in np.arange(number_class):
        index_train=np.where(Y_train==labels[i])
        index_test=np.where(Y_test==labels[i])
        y_train[i,index_train]=1 
        y_test[i,index_test]=1
    return x_train,x_test,y_train,y_test

def activation_function(mu):
    M=20
    return (1/M)*(tf.nn.softplus(tf.cast(M*(mu+1.0),tf.float64))-np.log(1.0+np.exp(M)))


def lambda_qut_sann_classification(
	  xsample,hat_p_training,nSample=100000,
      miniBatchSize=500,alpha=0.05,option='quantile'):
    if np.mod(nSample,miniBatchSize)==0:
    	offset=0
    else:
    	offset=1
        
    n,p1= xsample.shape 
    # xsample = np.concatenate((xsample, np.ones([n,1])),axis=1)
    # p1+=1
    # xsample=tf.nn.l2_normalize(tf.cast(xsample,tf.float64),0)
  
    number_class=len(hat_p_training)

    fullList = np.zeros((miniBatchSize*(nSample//miniBatchSize+offset),))

    for index in range(nSample//miniBatchSize+offset):
        ySample = np.random.multinomial(1,hat_p_training,size=(n,1,miniBatchSize))
        y_mean = np.mean(ySample,axis=0)

        xy = np.zeros(shape=(n,p1,miniBatchSize,number_class))
        for index_n in np.arange(n):
            xy[index_n,:,:,:]=np.outer(xsample[index_n,:],
              (y_mean-ySample)[index_n,:,:]).reshape((p1,
              miniBatchSize,number_class))
        xymax = np.amax(tf.reduce_sum(np.abs(tf.reduce_sum(xy,axis=0).numpy()),
          axis=2).numpy(),axis=0)

        x = tf.constant(0.)
        with tf.GradientTape() as g:
        	g.watch(x)
        	y=activation_function(x)
        dy_dx=g.gradient(y,x)

        stat=dy_dx * xymax

        fullList[index*miniBatchSize:(index+1)*miniBatchSize]= stat[:]
        
    if option=='full':
    	return fullList
    elif option=='quantile':
    	return np.quantile(fullList,1-alpha)
    else:
    	pass
    
    
def shrinkage_operator(u,lambda_):
    return np.sign(u) * np.maximum((np.abs(u) - lambda_), 0.)


def TFmodel(Xsample,Ysample,p2,activation_function,lamb,learningRate,
      withSD,ifTrain=True,iterationISTA=3000, Xtest=None, iniscale=0.01, 
      rel_err=1.e-9, w1_ini=None, w2_ini=None, c_ini=None):
    
    n,p1= Xsample.shape 
    number_class=Ysample.shape[0]
    
    if w1_ini is None:
        w1_ini = np.random.normal(loc=0., scale=iniscale, size=(p2,p1))
    if w2_ini is None:
        w2_ini =np.random.normal(loc=0., scale=iniscale, size=(number_class,p2))
    if c_ini is None:   
        c_ini = np.random.normal(loc=0., scale=iniscale, size=(number_class,1))
    
                   
    w1 = tf.Variable(w1_ini,shape=(p2,p1), trainable=True)
    w2 = tf.Variable(w2_ini,shape=(number_class,p2), trainable=True)
    c  = tf.Variable(c_ini,shape=(number_class,1), trainable=True)
    
    t=1        
    w1_y=tf.Variable(w1_ini,shape=(p2,p1), trainable=True)
    w2_y=tf.Variable(w2_ini,shape=(number_class,p2), trainable=True)
    c_y=tf.Variable(c_ini,shape=(number_class,1), trainable=True)
    
    def evaluate(examx, examy,ifUpdate=True, ifISTA=False):
        nonlocal w1
        nonlocal w2 
        nonlocal c 
        nonlocal w1_y
        nonlocal w2_y
        nonlocal c_y
        nonlocal t
        
        #############################################################################################
        # new structure in TF2.0 to track the parts of computation that will need backprop
        with tf.GradientTape(persistent=True) as g:
            g.watch([w1_y,w2_y,c_y])
            w2_l2normalized_1=tf.nn.l2_normalize(tf.cast(w2_y,tf.float64),1)
            mu_2layer_1=tf.matmul(w2_l2normalized_1,activation_function(tf.matmul(w1_y,tf.transpose(examx))))+tf.matmul(c_y,np.ones([1,n]))
            softmax_mu_2layer_1=tf.transpose(tf.nn.softmax(tf.transpose(mu_2layer_1)))
            bareCost= -tf.linalg.trace(tf.matmul(tf.transpose(tf.convert_to_tensor(examy)),tf.math.log(softmax_mu_2layer_1)))
            cost = bareCost + lamb*tf.reduce_sum(tf.abs(w1_y))
        
        if not (ifISTA):
            w1_gra= g.gradient(cost, w1_y)
        if (ifISTA):
            bare_w1_gra= g.gradient(bareCost, w1_y)
        w2_gra= g.gradient(cost, w2_y)
        c_gra = g.gradient(cost, c_y)           
        del g
        
        w1_value=w1.numpy()
        w2_value=w2.numpy()
        c_value=c.numpy()
        ############################################
        if (ifUpdate):
            if (ifISTA):
                proposed_w1 = shrinkage_operator(w1_y-bare_w1_gra*learningRate,lamb*learningRate)
                proposed_w2 = w2_y - learningRate*w2_gra  ###normalize
                proposed_c = c_y - learningRate*c_gra
            else:
                proposed_w1 = w1_y-learningRate*w1_gra
                proposed_w2 = w2_y - learningRate*w2_gra
                proposed_c = c_y - learningRate*c_gra
            
            w1.assign(proposed_w1)
            w2.assign(proposed_w2)
            c.assign(proposed_c)
            
            t_1=(1+np.sqrt(1+4*(t**2)))/2
            
            w1_atProposition = w1.numpy()
            w2_atProposition = w2.numpy()
            c_atProposition = c.numpy()
            
            w1_y=tf.Variable(w1_atProposition+(t-1)/t_1*(w1_atProposition-w1_value),shape=(p2,p1), trainable=True)
            w2_y=tf.Variable(w2_atProposition+(t-1)/t_1*(w2_atProposition-w2_value),shape=(number_class,p2), trainable=True)
            c_y=tf.Variable(c_atProposition+(t-1)/t_1*(c_atProposition-c_value),shape=(number_class,1), trainable=True)
       
            t=t_1

            w2_normalized_atProposition=tf.nn.l2_normalize(tf.cast(w2_atProposition,tf.float64),1).numpy()
            mu_2layer_atProposition=tf.matmul(w2_normalized_atProposition,activation_function(tf.matmul(w1_atProposition,tf.transpose(examx))))+tf.matmul(c_atProposition,np.ones([1,n]))
            softmax_mu_2layer_atProposition=tf.transpose(tf.nn.softmax(tf.transpose(mu_2layer_atProposition)))
            bareCost_atProposition= - tf.linalg.trace(tf.matmul(tf.transpose(tf.convert_to_tensor(examy)),tf.math.log(softmax_mu_2layer_atProposition)))
            cost_atProposition = bareCost_atProposition +lamb*tf.reduce_sum(tf.abs(w1_atProposition))
        return cost_atProposition 
    
    if (ifTrain):
        # iteratively update the model parameters
        bestcost=float('inf')
        cont=withSD #steepest descent
        epoch=0
        #for epoch in range(iterationSD):# steepest descent part
        while(cont):
            epoch +=1
            if epoch % 100 == 0 and epoch != 0:
                cost = evaluate(Xsample,Ysample,ifUpdate=True, ifISTA=False)
                print('         At epoch', epoch,'cost =', cost.numpy())
                if(cost<bestcost):
                    bestcost=cost
                else:
                    cont=False
                if(cost<1e-5):
                    cont=False        
            cost = evaluate(Xsample, Ysample, ifUpdate=True, ifISTA=False)   
        bestcost=float('inf')
        epoch=0
        cont=True
        while(cont):
            epoch += 1
        #for epoch in range(iterationISTA):# ISTA part           
            if epoch % 100 == 0 and epoch != 0:
                print('         At epoch FISTA', epoch,'cost =', cost.numpy())
                if(cost>bestcost):
                    learningRate *= .99
                    print(learningRate)
                if(np.abs(cost-bestcost)/cost<rel_err):
                    cont=False
                bestcost=cost
            cost = evaluate(Xsample, Ysample, ifUpdate=True, ifISTA=True)
            cont = (epoch<iterationISTA) & cont
    
    # we return only the numpy values
    w2_l2normalized=tf.nn.l2_normalize(tf.cast(w2,tf.float64),1)
    mu_2layer=tf.matmul(w2_l2normalized,activation_function(tf.matmul(w1,tf.transpose(Xsample))))+tf.matmul(c,np.ones([1,n]))
    p_hat=tf.transpose(tf.nn.softmax(tf.transpose(mu_2layer)))
    one_index=np.where(p_hat==np.max(p_hat,axis=0))
    y_hat=np.zeros([number_class,n])
    y_hat[one_index]=1

    if np.any(Xtest):
        ntest,p1test= Xtest.shape
        mu_2layer_test=tf.matmul(w2_l2normalized,activation_function(tf.matmul(w1,tf.transpose(Xtest))))+tf.matmul(c,np.ones([1,ntest]))
        p_test_hat=tf.transpose(tf.nn.softmax(tf.transpose(mu_2layer_test)))
        one_index_test=np.where(p_test_hat==np.max(p_test_hat,axis=0))
        y_test_hat=np.zeros([number_class,ntest])
        y_test_hat[one_index_test]=1
    else:
        p_test_hat = None
        y_test_hat = None
    return [w1,w2,c, p_hat, y_hat, p_test_hat,y_test_hat,cost,learningRate]

def TFmodelfit(X,y,p2,activation_function,lamb,learningRate0,withSD,ifTrain=True,iterationISTA=3000, Xtest=None, iniscale=0.01):
 
    lro=learningRate0[0]
    [w10,w20,c0,p_hat_0,y_hat_0,p_test_hat_0,y_test_hat_0,cost_0,lro_0] = TFmodel(X,y,p2,activation_function,lamb,lro,withSD=False,ifTrain=True,iterationISTA=-999,  Xtest=Xtest, iniscale=iniscale)
    [w1o,w2o,co,p_hat_o,y_hat_o,p_test_hat_o,y_test_hat_o,cost_o,lro_o] = TFmodel(X,y,p2,activation_function,lamb,lro_0,withSD,ifTrain=True,iterationISTA=iterationISTA,  Xtest=Xtest, iniscale=iniscale,w1_ini=w10, w2_ini=w20, c_ini=c0)
    
    for lr in learningRate0[1:]:
        [w1oo,w2oo,coo,p_hat_oo,y_hat_oo,p_test_hat_oo,y_test_hat_oo,cost_oo,lro_oo]= TFmodel(X,y,p2,activation_function,lamb,lr,withSD,ifTrain=True,iterationISTA=iterationISTA, Xtest=Xtest, iniscale=iniscale,w1_ini=w10, w2_ini=w20, c_ini=c0)
        if (cost_oo<cost_o):
            cost_o=cost_oo
            w1o=w1oo
            w2o=w2oo
            co=coo
            p_hat_o=p_hat_oo
            y_hat_o=y_hat_oo
            p_test_hat_o=p_test_hat_oo
            y_test_hat_o=y_test_hat_oo
            lro_o=lro_oo
        else:
            break
    return [w1o,w2o,co,p_hat_o,y_hat_o,p_test_hat_o,y_test_hat_o,cost_o,lro_o] 

def resultanalysis(w1o,y_test,y_test_hat_o,needles_index_true=None):
    p1=w1o.shape[1]
    needles_index_hat=np.array(np.where(np.sum(abs(w1o[:,0:(p1-1)]),axis=0)>0),dtype=float)+1
    if np.any(needles_index_true)!=None:
        
        s=len(needles_index_true)
        all_index=np.arange(1,p1)
        noneedles_index_true=np.delete(all_index,np.where(np.isin(all_index,needles_index_true)))
        
        tp_index=needles_index_hat[np.where(np.isin(needles_index_hat,needles_index_true)==True)]
        fp_index=needles_index_hat[np.where(np.isin(needles_index_hat,noneedles_index_true)==True)]
        tpr=len(tp_index)/s
        fdr=len(fp_index)/np.max([needles_index_hat.shape[1],1])
        
        hatintrue=np.where(np.isin(needles_index_hat,needles_index_true)==True)[1]
        trueinhat=np.where(np.isin(needles_index_true,needles_index_hat)==True)
        if np.all(hatintrue==trueinhat)&(len(hatintrue)!=0):#######
            hat_equal_true=True
        else:
            hat_equal_true=False
    else:
        tpr=None
        fdr=None
        hat_equal_true=None
    ################################################
    if np.any(y_test)!=None:
        number_class =y_test.shape[0]
        ###confusion :row-hat,colume-true
        confusion_matrix= np.zeros([number_class,number_class])
        #####assume the class is first,second,third,...
        number=np.sum(y_test,axis=1)
        for i in np.arange(0,number_class):
            for j in np.arange(0,number_class):
               confusion_matrix[i,j]=np.sum(y_test_hat_o[i,np.where(y_test[j,]==1)])
        accuracy=np.sum(np.diag(confusion_matrix))/np.sum(np.sum(y_test,axis=1))
    else:
        confusion_matrix=None
        accuracy=None
    return needles_index_hat,tpr,fdr,hat_equal_true,confusion_matrix,accuracy


def computeResults(x_train,y_train,learningRate_list,iniscale,lambda_qut,p2,x_test=None,y_test=None):
    ipath=-1
    lambi=np.exp(ipath)/(1+np.exp(ipath))*lambda_qut
    [w1o,w2o,co,p_hat_o,y_hat_o,p_test_hat_o,y_test_hat_o,cost_o,lro]=TFmodelfit(x_train,y_train,p2,activation_function,lambi,learningRate_list,True,ifTrain=True,iterationISTA=3000, Xtest=None, iniscale=iniscale)
    for ipath in np.arange(5):
        lambi=np.exp(ipath)/(1+np.exp(ipath))*lambda_qut
        [w1o,w2o,co,p_hat_o,y_hat_o,p_test_hat_o,y_test_hat_o,cost_o,lro]=TFmodel(x_train,y_train,p2,activation_function,lambi,lro,
          True,ifTrain=True,iterationISTA=3000, Xtest=None,
          rel_err=1.e-9, w1_ini=w1o, w2_ini=w2o, c_ini=co)
    if np.any(x_test)==None:
        [w1o,w2o,co,p_hat_o,y_hat_o,p_test_hat_o,y_test_hat_o,cost_o,lro]=TFmodel(x_train,y_train,p2,activation_function,lambda_qut,lro,
          False,ifTrain=True,iterationISTA=10000, Xtest=None, 
          rel_err=1.e-12, w1_ini=w1o, w2_ini=w2o, c_ini=co)
    else:
        [w1o,w2o,co,p_hat_o,y_hat_o,p_test_hat_o,y_test_hat_o,cost_o,lro]=TFmodel(x_train,y_train,p2,activation_function,lambda_qut,lro,
          False,ifTrain=True,iterationISTA=10000, Xtest=x_test, iniscale=0.0, 
          rel_err=1.e-12, w1_ini=w1o, w2_ini=w2o, c_ini=co)
    needles_index_hat,tpr,fdr,hat_equal_true,confusion_matrix,accuracy=resultanalysis(w1o,y_test,y_test_hat_o)
    return w1o,w2o,co,p_hat_o,y_hat_o,p_test_hat_o,y_test_hat_o,cost_o,lro,needles_index_hat,tpr,fdr,hat_equal_true,confusion_matrix,accuracy

def writeResults(filename,p2,nSample,lambda_qut,cost_o,confusion_matrix,y_test,accuracy,needles_index_hat,w1o,w2o,co):
    with open(filename,'a') as file_object:
        file_object.write("Results for p2=" + str(p2)+" lambdaqut_nSample= "+str(nSample)+"\n")
        file_object.write("\n")
        file_object.write("lambda_qut="+str(lambda_qut)+"\n")
        file_object.write("\n")
        file_object.write("cost="+str(cost_o)+"\n")
        file_object.write("\n")
        file_object.write("confusion_matrix"+"\n")
        file_object.write(str(confusion_matrix)+"\n")
        file_object.write("\n")
        file_object.write("True test sample size for every class:"+"\n")
        file_object.write(str(np.sum(y_test,axis=1))+"\n")
        file_object.write("\n")
        file_object.write("Accuracy:"+"\n")
        file_object.write(str(round(accuracy*100,2))+"%"+"\n")
        file_object.write("\n")
        file_object.write(str(needles_index_hat.shape[1])+" needles"+"\n")
        file_object.write("needles_hat:"+"\n")
        file_object.write(str(needles_index_hat)+"\n")
        file_object.write("\n")
        file_object.write("Sum of rows for np.abs(w1):"+"\n")
        file_object.write(str(np.sum(np.abs(w1o),axis=1))+"\n")
        file_object.write("\n")
        file_object.write("w2:"+"\n")
        file_object.write(str(w2o)+"\n")
        file_object.write("\n")
        file_object.write("c:"+"\n")
        file_object.write(str(co)+"\n")






