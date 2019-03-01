#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 19:15:45 2018

@author: junewang
"""
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

station_label=pd.read_csv("station_label.csv",header=None)
aiqQuality_201804=pd.read_csv("aiqQuality_201804.csv")
airQuality_201701_201801=pd.read_csv("airQuality_201701-201801.csv")
airQuality_201802_201803=pd.read_csv("airQuality_201802-201803.csv")
Beijing_grid_weather_station=pd.read_csv("Beijing_grid_weather_station.csv",header=None)
Beijing_grid_weather_station.columns = ['station_id', 'longitude', 'latitude']
gridWeather_201701_201803=pd.read_csv("gridWeather_201701-201803.csv")
gridWeather_201804=pd.read_csv("gridWeather_201804.csv")
gridWeather_20180501_20180502=pd.read_csv("gridWeather_20180501-20180502.csv")
observedWeather_201701_201801=pd.read_csv("observedWeather_201701-201801.csv")
observedWeather_201802_201803=pd.read_csv("observedWeather_201802-201803.csv")
observedWeather_201804=pd.read_csv("observedWeather_201804.csv")
observedWeather_20180501_20180502=pd.read_csv("observedWeather_20180501-20180502.csv")
aiqQuality_201804=pd.read_csv("aiqQuality_201804.csv")
station_info=pd.read_csv("station.csv",header=None)
station_info.columns = ['station_id', 'longitude', 'latitude']
station=list(station_info["station_id"])
##### 更改列明
station_label.columns = ['station_id', 'longitude', 'latitude','station_type']
del station_label["longitude"]
del station_label["latitude"]
aiqQuality_201804.columns = ['id', 'station_id', 'utc_time','PM2.5','PM10','NO2','CO','O3','SO2']
airQuality_201701_201801.rename(columns={ airQuality_201701_201801.columns[0]: "station_id" }, inplace=True)
airQuality_201802_201803.rename(columns={ airQuality_201802_201803.columns[0]: "station_id" }, inplace=True)
aiqQuality_201804=aiqQuality_201804.drop(['id'], axis=1)

gridWeather_201701_201803.rename(columns={ gridWeather_201701_201803.columns[0]: "station_id" }, inplace=True)
gridWeather_201701_201803.rename(columns={ gridWeather_201701_201803.columns[0]: "station_id" }, inplace=True)
gridWeather_201701_201803.rename(columns={ gridWeather_201701_201803.columns[-1]: "wind_speed" }, inplace=True)
gridWeather_201804.rename(columns={ gridWeather_201804.columns[2]: "utc_time" }, inplace=True)
gridWeather_20180501_20180502.rename(columns={ gridWeather_20180501_20180502.columns[2]: "utc_time" }, inplace=True)
gridWeather_201804=gridWeather_201804.drop(['id'],axis=1)
gridWeather_201701_201803=gridWeather_201701_201803.drop(['longitude','latitude'],axis=1)
gridWeather_20180501_20180502=gridWeather_20180501_20180502.drop(['id'],axis=1)

observedWeather_201804.rename(columns={ observedWeather_201804.columns[2]: "utc_time" }, inplace=True)
observedWeather_20180501_20180502.rename(columns={ observedWeather_20180501_20180502.columns[2]: "utc_time" }, inplace=True)
observedWeather_201701_201801=observedWeather_201701_201801.drop(['longitude','latitude'],axis=1)
observedWeather_201804=observedWeather_201804.drop(['id',],axis=1)
observedWeather_20180501_20180502=observedWeather_20180501_20180502.drop(['id',],axis=1)

concat_airqQuality=pd.concat([aiqQuality_201804,airQuality_201701_201801,airQuality_201802_201803])
concat_gridWeather=pd.concat([gridWeather_201804,gridWeather_201701_201803,gridWeather_20180501_20180502])
concat_observedWeather=pd.concat([observedWeather_201701_201801,observedWeather_201802_201803,observedWeather_201804,observedWeather_20180501_20180502])

concat_airqQuality.sort_values(by=["station_id","utc_time"], inplace=True)
concat_gridWeather.sort_values(by=["station_id","utc_time"], inplace=True)
concat_observedWeather.sort_values(by=["station_id","utc_time"], inplace=True)

concat_airqQuality=concat_airqQuality.drop_duplicates(keep='first')
concat_gridWeather=concat_gridWeather.drop_duplicates(keep='first')
concat_observedWeather=concat_observedWeather.drop_duplicates(keep='first')


concat_airqQuality["time_month"]=concat_airqQuality["utc_time"].apply(lambda x:time.strptime(str(x), "%Y-%m-%d %H:%M:%S")[1])
concat_airqQuality=concat_airqQuality[concat_airqQuality["time_month"].isin([3,4,5,6])]

concat_gridWeather["time_month"]=concat_gridWeather["utc_time"].apply(lambda x:time.strptime(str(x), "%Y-%m-%d %H:%M:%S")[1])
concat_gridWeather=concat_gridWeather[concat_gridWeather["time_month"].isin([3,4,5,6])]

concat_observedWeather["time_month"]=concat_observedWeather["utc_time"].apply(lambda x:time.strptime(str(x), "%Y-%m-%d %H:%M:%S")[1])
concat_observedWeather=concat_observedWeather[concat_observedWeather["time_month"].isin([3,4,5,6])]


unique_station_gridWeather=concat_gridWeather["station_id"].unique()
unique_station_airQuality=list(concat_airqQuality["station_id"].unique())
unique_station_observedWeather=pd.DataFrame(concat_observedWeather["station_id"].unique())
concat_gridWeather=pd.merge(concat_gridWeather,Beijing_grid_weather_station,on="station_id")
concat_airqQuality=pd.merge(concat_airqQuality,station_info,on="station_id")

concat_airqQuality.info()
concat_gridWeather.info()
concat_observedWeather.info()

#concat_airqQuality.to_csv("process_data/concat_airqQuality.csv")
#concat_gridWeather.to_csv("process_data/concat_gridWeather.csv")
#concat_observedWeather.to_csv("process_data/concat_observedWeather.csv")

###plot, the location of the station
plt.scatter(station_info["longitude"],station_info["latitude"])
plt.title('station location distribution')

###missing value
def get_missing_value_number(unique_station_airQuality,concat_airqQuality,col):
    missing=pd.DataFrame(columns=["stations","missing_number","missing_ratio"])
    for i in range(len(unique_station_airQuality)):
        df=concat_airqQuality[concat_airqQuality["station_id"]==unique_station_airQuality[i]]
        missing_number=len(df)-df[col].count()
        missing_ratio=missing_number/len(df)
        dic={"stations":unique_station_airQuality[i],"missing_number":missing_number,"missing_ratio":missing_ratio}
        missing=missing.append(dic,ignore_index=True)
    return missing
missing=get_missing_value_number(unique_station_airQuality,concat_airqQuality,"PM10")        
  
#concat_airqQuality=pd.read_csv("process/concat_airqQuality.csv")
#concat_gridWeather=pd.read_csv("process/concat_gridWeather.csv")
#concat_observedWeather=pd.read_csv("process/concat_observedWeather.csv")
#concat_airqQuality=concat_airqQuality.iloc[:,1:]

###1. interpolated method1:spatial interpolation
def get_distance_matrix(station_info):
    distance_matrix=np.zeros([len(station_info),len(station_info)])
    for i in range(len(station_info)):
        for j in range(len(station_info)):
            dist=pow((station_info.iloc[i,1]-station_info.iloc[j,1]),2)+pow((station_info.iloc[i,2]-station_info.iloc[j,2]),2)
            distance_matrix[i,j]=np.sqrt(dist)
    return distance_matrix
distance_matrix=get_distance_matrix(station_info)

def get_3_means_station(distance_matrix,unique_station_airQuality):
    dic={}
    for i in range(len(unique_station_airQuality)):
        station_tuple=zip(unique_station_airQuality,list(distance_matrix[:,i]))
        station_tuple=sorted(station_tuple,key=lambda t:(-t[1],t[0]))
        dic[unique_station_airQuality[i]]=[station_tuple[i] for i in range(3)]
    return dic
k_means_dic=get_3_means_station(distance_matrix,unique_station_airQuality)
        
###空间插值：IDW算法插值，距离越远，占的比重越小
def interpolate_attri(k_means_dic,concat_airqQuality,attri):
    def get_weight(k_means_dic):
        dicx={}
        for k,v in k_means_dic.items():
            sum_dist=sum([v[i][1] for i in range(3)])
            p=[]
            for x in v:
                p.append(x[1]/sum_dist)
            sum_p=sum(p)
            for i in range(len(p)):
                p[i]=(v[i][0],round(p[i]/sum_p,3))
            dicx[k]=p
        return dicx
    dicx=get_weight(k_means_dic)
    dic_station={}
    dic_weight={}
    for k,v in dicx.items():
        dic_station[k]=[x[0] for x in v]
        dic_weight[k]=[x[1] for x in v]

    for i in range(len(concat_airqQuality)):
        print(i)
        if np.isnan(concat_airqQuality.loc[i,attri])==True:
            print(i)
            df=concat_airqQuality[concat_airqQuality["utc_time"]==concat_airqQuality.loc[i,"utc_time"]]
            df=df[df["station_id"].isin(dic_station[concat_airqQuality.loc[i,"station_id"]])]
            pre_val=0
            df=df.reset_index()
            del df['index']
            stations=dic_station[concat_airqQuality.loc[i,"station_id"]]
            weights=dic_weight[concat_airqQuality.loc[i,"station_id"]]
            for j in range(len(df)):
                if np.isnan(df.loc[j,attri])==True:
                    continue
                index=stations.index(df.iloc[j,0])
                pre_val+=df.loc[j,attri]*weights[index]
            if pre_val!=0:
                concat_airqQuality.loc[i,attri]=pre_val

    return concat_airqQuality
    
concat_airqQuality=interpolate_attri(k_means_dic,concat_airqQuality,"PM10")
concat_airqQuality=interpolate_attri(k_means_dic,concat_airqQuality,"CO")
concat_airqQuality=interpolate_attri(k_means_dic,concat_airqQuality,"NO2")
concat_airqQuality=interpolate_attri(k_means_dic,concat_airqQuality,"O3")
concat_airqQuality=interpolate_attri(k_means_dic,concat_airqQuality,"PM2.5")
concat_airqQuality=interpolate_attri(k_means_dic,concat_airqQuality,"SO2")

####2. interpolated method2: 利用arima模型预测插值
concat_airqQuality_inpl=concat_airqQuality
stations=list(concat_airqQuality_inpl["station_id"].unique())

def ARMA_MODEL(timeseries): #返回预测值
    order=sm.tsa.arma_order_select_ic(timeseries,max_ar=3,max_ma=3,ic='bic')['bic_min_order'] 
    temp_model= sm.tsa.ARMA(timeseries,order).fit()
    pred=temp_model.forecast(1) #返回后面一期
    return pred[0][-1]

def Arima_interpolate(air_qual,attri,stations):
    air_qual_pre=pd.DataFrame()
    for station in stations:
        print(station)
        temp=air_qual[air_qual["station_id"]==station]
        temp=temp.reset_index()
        del temp['index']
        for i in range(len(temp)):
            if np.isnan(temp.loc[i,attri])==True:
                print(i)
                timeseries=temp[attri][:i]
                try:
                    temp.loc[i,attri]=ARMA_MODEL(timeseries)
                except:
                    print("not predit")
        air_qual_pre=pd.concat([air_qual_pre,temp],axis=0)
    return air_qual_pre

concat_airqQuality_inpl=Arima_interpolate(concat_airqQuality_inpl,"PM2.5",stations)
concat_airqQuality_inpl=Arima_interpolate(concat_airqQuality_inpl,"O3",stations)

concat_airqQuality_inpl=Arima_interpolate(concat_airqQuality_inpl,"SO2",stations)
concat_airqQuality_inpl=Arima_interpolate(concat_airqQuality_inpl,"NO2",stations)    


###3. interpolated method3: 时间差值：
def interpolate(df,stations):
    res_df=pd.DataFrame()
    for station in stations:
        dfx=df[df["station_id"]==station]
        dfx=dfx.interpolate(method="cubic")
        res_df=res_df.append(dfx)
    return res_df
  
concat_airqQuality_inpl=interpolate(concat_airqQuality_inpl,stations)

concat_airqQuality_inpl.to_csv("process_data/concat_airqQuality_all_interpolated.csv")
























