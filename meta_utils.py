import numpy as np
import pandas as pd
from lag import make_lag,data_shot,cvs
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)
pd.set_option('display.max_colwidth',1000)

data_max=data_shot['cv1'][1]
print(data_max)
cv_train=[]
cv_test=[]
for i in range(8):
    cv_train.append(f'cv{i+1}')
for i in range(8,12):
    cv_test.append(f'cv{i+1}')
print(cv_train)
print(cv_test)
def get_prediction(pool_type,h):
    arima_result=pd.read_csv('/Users/zhongming/Desktop/245天/arima_result.csv', index_col=0)
    svm_result_3 = pd.read_csv('/Users/zhongming/Desktop/245天/svm_result_3.csv', index_col=0)
    svm_result_1 = pd.read_csv('/Users/zhongming/Desktop/245天/svm_result_1.csv', index_col=0)
    ridge_result_3 = pd.read_csv('/Users/zhongming/Desktop/245天/ridge_result_3.csv', index_col=0)
    ridge_result_1 = pd.read_csv('/Users/zhongming/Desktop/245天/ridge_result_1.csv', index_col=0)
    lgb_result_7 = pd.read_csv('/Users/zhongming/Desktop/245天/lgb_result_7.csv', index_col=0)
    lgb_result_3 = pd.read_csv('/Users/zhongming/Desktop/245天/lgb_result_3.csv', index_col=0)
    rf_result_7 = pd.read_csv('/Users/zhongming/Desktop/245天/rf_result_7.csv', index_col=0)
    rf_result_3 = pd.read_csv('/Users/zhongming/Desktop/245天/rf_result_3.csv', index_col=0)
    ridgep_result_3 = pd.read_csv('/Users/zhongming/Desktop/245天/ridgep_result_3.csv', index_col=0)
    ridgep_result_7 = pd.read_csv('/Users/zhongming/Desktop/245天/ridgep_result_7.csv', index_col=0)
    models_individual=pd.concat([arima_result,svm_result_1,svm_result_3,ridge_result_1,ridge_result_3],axis=1)
    models_pool=pd.concat([ridgep_result_3,ridgep_result_7,rf_result_3,rf_result_7,lgb_result_3,lgb_result_7],axis=1)
    models_mixed=pd.concat([arima_result,svm_result_1,ridge_result_1,ridgep_result_7,rf_result_7,lgb_result_7],axis=1)
    nsample_train=134*8
    nsample_test=134*4
    if pool_type=='mixed':
        pred_train=models_mixed.iloc[:nsample_train,:].values.reshape(nsample_train,-1,h)
        pred_test=models_mixed.iloc[:nsample_test,:].values.reshape(nsample_test,-1,h)

    if pool_type=='pooling':
        pred_train = models_pool.iloc[:nsample_train, :].values.reshape(nsample_train, -1, h)
        pred_test = models_pool.iloc[:nsample_test, :].values.reshape(nsample_test, -1, h)


    if pool_type=='individual':
        pred_train = models_individual.iloc[:nsample_train, :].values.reshape(nsample_train, -1, h)
        pred_test = models_individual.iloc[:nsample_test, :].values.reshape(nsample_test, -1, h)


    return pred_train,pred_test

def reshape_X(X,n):
    k=int(X.shape[1]/n)
    Y=np.zeros(shape=(X.shape[0],k,n))
    for i in range(n):
        Y[:,:,i]=X[:,i*k:(i+1)*k]
    return Y

def data_metalearning_prepare(data,h):
    X_train_y = pd.DataFrame()
    Y_train = pd.DataFrame()
    X_train_p = pd.DataFrame()
    X_test_y = pd.DataFrame()
    X_test_p = pd.DataFrame()
    Y_test = pd.DataFrame()
    for cv in cv_train:
        Y_columns = ['TurbID', 'Day', 'Patv']
        feature_columns = ['TurbID', 'Day', 'Wspd', 'Itmp', 'Etmp', 'Wdir', 'Ndir', 'Pab1', 'Pab2', 'Pab3']
        feature_columns_1 = ['Wspd', 'Itmp', 'Etmp', 'Wdir', 'Ndir', 'Pab1', 'Pab2', 'Pab3']
        data_mask = (data.Day >= data_shot[cv][0]) & (data.Day <= data_shot[cv][1])
        df = data[data_mask]
        x_train_y_mask = (df.Day >= data_shot[cv][0]) & (df.Day <= data_shot[cv][1] - h)
        y_train_mask = df.Day > data_shot[cv][1] - h
        Y_train = pd.concat([Y_train,pd.DataFrame(df[y_train_mask][Y_columns].pivot_table(index='TurbID', columns='Day').values,columns=[str(i) + 'day' for i in range(1, 8)])])
        X_train_y=pd.concat([X_train_y,pd.DataFrame(df[x_train_y_mask][Y_columns].pivot_table(index='TurbID', columns='Day').values,columns=[str(i) + 'day' for i in range(1, data_max+1-h)])])
        X_train_p = pd.concat([X_train_p, pd.DataFrame(df[feature_columns].pivot_table(index='TurbID', columns='Day')[feature_columns_1].values,columns=[str(i) + 'day' for i in range(1, data_max + 1)] * 8)])


    for cv in cv_test:
        Y_columns = ['TurbID', 'Day', 'Patv']
        feature_columns = ['TurbID', 'Day', 'Wspd', 'Itmp', 'Etmp', 'Wdir', 'Ndir', 'Pab1', 'Pab2', 'Pab3']
        feature_columns_1 = ['Wspd', 'Itmp', 'Etmp', 'Wdir', 'Ndir', 'Pab1', 'Pab2', 'Pab3']
        data_mask = (data.Day >= data_shot[cv][0]) & (data.Day <= data_shot[cv][1])
        df = data[data_mask]
        x_test_y_mask = (df.Day >= data_shot[cv][0]) & (df.Day <= data_shot[cv][1] - h)
        y_test_mask = df.Day > data_shot[cv][1] - h
        Y_test = pd.concat([Y_test,pd.DataFrame(df[y_test_mask][Y_columns].pivot_table(index='TurbID', columns='Day').values,columns=[str(i) + 'day' for i in range(1, 8)])])
        X_test_y = pd.concat([X_test_y, pd.DataFrame(df[x_test_y_mask][Y_columns].pivot_table(index='TurbID', columns='Day').values,columns=[str(i) + 'day' for i in range(1, data_max + 1 - h)])])
        X_test_p = pd.concat([X_test_p, pd.DataFrame(df[feature_columns].pivot_table(index='TurbID', columns='Day')[feature_columns_1].values,columns=[str(i) + 'day' for i in range(1, data_max + 1)] * 8)])# 有8个特征
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler_y= MinMaxScaler()
    scaler_x= StandardScaler()
    X_train_y = reshape_X(scaler_y.fit_transform(X_train_y.values), 1)
    X_train_p = reshape_X(scaler_x.fit_transform(X_train_p),8)

    X_test_y = reshape_X(scaler_y.fit_transform(X_test_y.values), 1)
    X_test_p = reshape_X(scaler_x.fit_transform(X_test_p), 8)
    return  Y_train,Y_test, X_train_y, X_train_p, X_test_y, X_test_p


def get_scaler(pred,true):
    scaler=np.zeros(shape=(true.shape[0],pred.shape[1]))
    for i in range(pred.shape[1]):
        scaler[:,i]=np.mean(np.square(pred[:,i,:]-true),axis=1)
    return np.mean(scaler,axis=1)
# Y_train,Y_test, X_train_y, X_train_p, X_test_y, X_test_p=data_metalearning_prepare(data,7)
#
# Y_train.to_csv('/Users/zhongming/Desktop/245天/Y_train.csv')
# Y_test.to_csv('/Users/zhongming/Desktop/245天/Y_test.csv')


# data=pd.read_csv('/Users/zhongming/Desktop/245天/last.csv',index_col=0)
#
# Y_train,Y_test,X_train_y,X_train_p,X_test_y,X_test_p=data_metalearning_prepare(data,7)
# print(Y_train.shape)
# base_train,base_test=get_prediction('mixed',7)
# print(get_scaler(base_train,Y_train))