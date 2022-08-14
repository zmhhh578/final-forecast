import numpy as np
import pandas as pd

data=pd.read_csv('/Users/zhongming/Desktop/245天/wtbdata_245days.csv')
def abnormal_mask(df):
    abnormals_1 = (df['Wspd'] > 2.5) & (df['Patv'] <= 0)
    abnormals_2 = df[['Pab1', 'Pab2', 'Pab3']].max(1) > 89
    abnormals_3 = (df['Ndir'] < -720) | (df['Ndir'] > 720)
    abnormals_4 = (df['Wdir'] < -180) | (df['Wdir'] > 180)

    missing_values = df.isna().mean(1).astype(bool) # mask all missing values

    abnormals = abnormals_1 | abnormals_2 | abnormals_3 | abnormals_4 | missing_values

    return abnormals
data_new=data.mask(abnormal_mask(data))
data_new=data_new.interpolate().fillna(method='bfill')
data_new.drop(["Prtv","Tmstamp"],axis=1,inplace=True)
data_new['Patv'].clip(lower=0,inplace=True)
data_new[['TurbID','Day']]=data_new[['TurbID','Day']].astype('int')
dff=data_new.pivot_table(index="TurbID",columns="Day",aggfunc='mean').stack().reset_index()
print(dff.isnull().sum())
print(dff)
dff=dff[['TurbID','Day','Patv','Wspd','Itmp','Etmp','Wdir','Ndir','Pab1','Pab2','Pab3']].to_csv('/Users/zhongming/Desktop/245天/last.csv')