import pandas as pd
import numpy as np
# data=pd.read_csv('/Users/zhongming/Desktop/245天/last.csv',index_col=0)
# print(data)

cvs=[]
h=7


for i in range(12):
    cvs.append(f'cv{i+1}')
data_shot={}
data_valid_shot={}
for i in range(12):
    data_shot[f'cv{i+1}']=[i*7+1,i*7+168]
print(data_shot)
#'Patv','Wspd','Itmp','Etmp','Wdir','Ndir','Pab1','Pab2','Pab3'
def make_lag(data,h,lag):
    patv_lag=data[['TurbID','Day','Patv']]
    for i in range(1,lag+h+1):
        patv_lag[f'Patv_lag{i}']=patv_lag.groupby("TurbID")['Patv'].transform(lambda x:x.shift(i,fill_value=x.iloc[0]))
    wspd_lag = data[['TurbID', 'Day', 'Wspd']]
    for i in range(1,lag+1):
        wspd_lag[f'Wspd_lag{i}']=wspd_lag.groupby("TurbID")['Wspd'].transform(lambda x:x.shift(i,fill_value=x.iloc[0]))
    # Itmp_lag = data[['TurbID', 'Day', 'Itmp']]
    # for i in range(1, lag + 1):
    #     Itmp_lag[f'Itmp_lag{i}'] = Itmp_lag.groupby("TurbID")['Itmp'].transform(lambda x: x.shift(i, fill_value=x.iloc[0]))
    return pd.concat([patv_lag,wspd_lag.iloc[:,2:],data[['Itmp','Wdir','Ndir','Pab1','Pab2','Pab3']]],axis=1)




