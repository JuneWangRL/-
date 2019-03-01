#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:59:32 2018

@author: junewang
"""
import pandas as pd
import datetime
import numpy as np
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt

concat_airqQuality_inpl=pd.read_csv("concat_airqQuality_feature_done.csv")
concat_airqQuality_inpl=concat_airqQuality_inpl.iloc[:,1:]

predict_df=pd.read_csv("process_data/predictx.csv")
predict_df=predict_df.iloc[:,1:]

###1.时间序列预测airquality数据
def station_test(ts):    #检验平稳性
    dftest=adfuller(ts,maxlag=10)
    df_p=dftest[1]
    if df_p>=0.05:
        stationarity=False
    elif df_p<0.05:
        stationarity=True
    return stationarity
#l=station_test(station0["SO2"])

def ARMA_MODEL(timeseries): #返回预测值
    order=sm.tsa.arma_order_select_ic(timeseries,max_ar=3,max_ma=3,ic='bic')['bic_min_order'] 
    temp_model= sm.tsa.ARMA(timeseries,order).fit()
    pred=temp_model.forecast(1) #返回后面一期
    return pred[0][-1]
#l1=ARMA_MODEL(station0["PM10"])
def decompose2(timeseries): #差分
    station_diff=False
    diff1=timeseries.diff(1)
    station_diff1=station_test(diff1)
    if station_diff1:
        pred=ARMA_MODEL(diff1)+timeseries.values[-1]
        station_diff=True
    else:      
        diff4=diff1.diff(4).dropna()
        station_diff4=station_test(diff4)
        if station_diff4:
            pred=ARMA_MODEL(diff4) + timeseries.values[-1] + timeseries.values[-4] - timeseries.values[-5]
            station_diff=True
        else:
            pred=0
            station_diff=False
    return station_diff,pred

def ARIMA_pred(air_qual,attri):
    stations=list(air_qual["station_id"].unique())
    air_qual=air_qual[air_qual["year"]>2017]
    air_qual=air_qual.sort_values("utc_time")
    pre_df=pd.DataFrame()
    time_span=[]
    time="2018-05-01 0:0:0"
    time=datetime.datetime.strptime(time,"%Y-%m-%d %H:%M:%S")
    for i in range(48):
        time=time+datetime.timedelta(hours = 1)
        time_span.append(time)
    #print(time_span)
    for i in range(len(stations)):
        print(i)
        t=pd.DataFrame(columns={"station_id","utc_time",attri})
        temp=air_qual[air_qual["station_id"]==stations[i]]
        temp=temp.fillna(method='ffill')
        timeseries=list(temp[attri])
        for j in range(48):
            timeseries=timeseries[-500:]
            stationarity=station_test(timeseries)
            if stationarity:
                try:
                    pre=ARMA_MODEL(timeseries)
                    timeseries.append(pre)  
                except:
                    timeseries.append(timeseries[-1]) 
                stat=True
            else:
                try:
                    stat,pred=decompose2(timeseries)
                except:
                    print("No")
            t.loc[j] = {"station_id":stations[i],"utc_time":time_span[j],attri:pre}
        pre_df=pd.concat([pre_df,t],axis=0)
    return pre_df


concat_airqQuality_inpl["year"]=concat_airqQuality_inpl["utc_time"].apply(lambda x:int(x[:4]))


pre_df4=ARIMA_pred(concat_airqQuality_inpl,"CO")
pre_df4.to_csv("process_data/pre_CO.csv")
pre_df5=ARIMA_pred(concat_airqQuality_inpl,"SO2")
pre_df6=ARIMA_pred(concat_airqQuality_inpl,"NO2")
pre_df5.to_csv("process_data/pre_SO2.csv")
pre_df6.to_csv("process_data/pre_NO2.csv")



pre_CO=pd.read_csv("process_data/pre_CO.csv")
pre_SO2=pd.read_csv("process_data/pre_SO2.csv")
pre_NO2=pd.read_csv("process_data/pre_NO2.csv")

predict_df=predict_df.sort_values(by=['station_id','utc_time'])
pre_CO=pre_CO.iloc[:,1:]
pre_SO2=pre_SO2.iloc[:,1:]
pre_NO2=pre_NO2.iloc[:,1:]
pre_CO["utc_time"]=pre_CO["utc_time"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S")-datetime.timedelta(hours = 1))
pre_SO2["utc_time"]=pre_SO2["utc_time"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S")-datetime.timedelta(hours = 1))
pre_NO2["utc_time"]=pre_NO2["utc_time"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S")-datetime.timedelta(hours = 1))
pre_CO=pre_CO.sort_values(by=['station_id','utc_time'])
pre_SO2=pre_SO2.sort_values(by=['station_id','utc_time'])
pre_NO2=pre_NO2.sort_values(by=['station_id','utc_time'])
predict_df["CO"]=pre_CO["CO"]
predict_df["SO2"]=pre_SO2["SO2"]
predict_df["NO2"]=pre_NO2["NO2"]
predict_df.to_csv("process_data/pre_df_done.csv")



###plot time series model
pre_df=pd.read_csv("process_data/pre_df_done.csv")
pre_df=pre_df.iloc[:,1:]
pre_df["WIND"]=0
pre_df["SNOW"]=0

test=pre_df[["station_id","utc_time","CO","SO2","NO2"]]
test["label"]=1
test=test[test["station_id"]=="aotizhongxin_aq"]

concat=concat_airqQuality_inpl[concat_airqQuality_inpl["year"]>2017]
concat=concat[concat["station_id"]=="aotizhongxin_aq"]
concat=concat.iloc[900:,:]
test2=concat[["station_id","utc_time","CO","SO2","NO2"]]
test2["label"]=0

testx=pd.concat([test2,test],axis=0)


plt.figure()
plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.legend(["CO"])
plt.xlabel('time')
plt.ylabel('CO(ug/m3)')
c=["r","b"]

O32 = testx[testx['label'] == 0]
plt.plot(range(len(testx)),testx["CO"],color=c[1])
O3 = testx[testx['label'] == 0]
plt.plot(range(len(O3)),O3["CO"],color=c[0])

plt.subplot(3,1,2)
plt.legend(["NO2"])
plt.xlabel('time')
plt.ylabel('NO2(ug/m3)')
c=["r","b"]

O32 = testx[testx['label'] == 0]
plt.plot(range(len(testx)),testx["NO2"],color=c[1])
O3 = testx[testx['label'] == 0]
plt.plot(range(len(O3)),O3["NO2"],color=c[0])


plt.subplot(3,1,3)
plt.legend(["SO2"])
plt.xlabel('time')
plt.ylabel('SO2(ug/m3)')
c=["r","b"]

O32 = testx[testx['label'] == 0]
plt.plot(range(len(testx)),testx["SO2"],color=c[1])
O3 = testx[testx['label'] == 0]
plt.plot(range(len(O3)),O3["SO2"],color=c[0])


















