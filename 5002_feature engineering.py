#!/usr/bin/env python3r=0.08

import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series
import time
import datetime
import math
import numpy as np

from pandas import Series
import time
import math
import datetime
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

station_label=pd.read_csv("station_label.csv",header=None)
station_label.columns = ['station_id', 'longitude', 'latitude','station_type']
del station_label["longitude"]
del station_label["latitude"]

concat_airqQuality_inpl=pd.read_csv("concat_airqQuality_all_interpolated.csv")
concat_airqQuality_inpl=concat_airqQuality_inpl.iloc[:,1:]
del concat_airqQuality_inpl["longitude"]
del concat_airqQuality_inpl["latitude"]

station_info=pd.read_csv("station.csv",header=None)
station_info.columns = ['station_id', 'longitude', 'latitude']

Beijing_grid_weather_station=pd.read_csv("Beijing_grid_weather_station.csv",header=None)
Beijing_grid_weather_station.columns = ['station_id', 'longitude', 'latitude']

stations=list(concat_airqQuality_inpl["station_id"].unique())
##### 更改列明
concat_gridWeather=pd.read_csv("concat_gridWeather.csv")
concat_gridWeather=concat_gridWeather.iloc[:,1:]
###station location picture
plt.scatter(station_info["longitude"],station_info["latitude"],color='b')
plt.scatter(Beijing_grid_weather_station["latitude"],Beijing_grid_weather_station["longitude"],s=5)
plt.title('station location distribution')

   
###get picture of air quality
p=concat_airqQuality_inpl[concat_airqQuality_inpl["station_id"]==stations[0]] 
temp=pd.DataFrame()
temp["time"]=p["utc_time"]
temp["O3"]=p["O3"]
temp["PM10"]=p["PM10"]
temp["PM2.5"]=p["PM2.5"]
temp.plot()

concat_airqQuality_inpl.max() 
concat_airqQuality_inpl.min() 
concat_gridWeather.min()

###站点配对
r=0.05
station_info["longitude_max"]=station_info["longitude"]+r
station_info["latitude_max"]=station_info["latitude"]+r
station_info["longitude_min"]=station_info["longitude"]-r
station_info["latitude_min"]=station_info["latitude"]-r


dic={}
for j in range(len(station_info)):
    dic[station_info.iloc[j,0]]=[]
    for i in range(len(Beijing_grid_weather_station)):
        if Beijing_grid_weather_station.iloc[i,2]>station_info.iloc[j,5]:
            if Beijing_grid_weather_station.iloc[i,2]<station_info.iloc[j,3]:
                if Beijing_grid_weather_station.iloc[i,1]>station_info.iloc[j,6]:
                    if Beijing_grid_weather_station.iloc[i,1]<station_info.iloc[j,4]:
                        dic[station_info.iloc[j,0]].append(Beijing_grid_weather_station.iloc[i,0])

max_len=0
for key,value in dic.items():
    if len(value)>max_len:
        max_len=len(value)
k=[]
v=[]
for key,value in dic.items():
    for val in value:
        k.append(key)
        v.append(val)
temp=pd.DataFrame(data={"grid_station":v,"station_id":k})    
sv=list(set(v))


###风速和风向分解到u,v方向
###1.direction feature
concat_gridWeather["direction_x"]=concat_gridWeather["wind_direction"].apply(lambda x:math.sin(math.radians(x)))
concat_gridWeather["direction_y"]=concat_gridWeather["wind_direction"].apply(lambda x:math.cos(math.radians(x)))

concat_gridWeather["speed_x"]=concat_gridWeather["wind_speed"]*concat_gridWeather["direction_x"]
concat_gridWeather["speed_y"]=concat_gridWeather["wind_speed"]*concat_gridWeather["direction_y"]
concat_gridWeather_use=concat_gridWeather[concat_gridWeather.station_id.isin(sv)]
concat_gridWeather_use = concat_gridWeather_use.reset_index()
concat_gridWeather_use=concat_gridWeather_use.drop(['index','direction_x','direction_y'],axis=1)

## 画图
stat=list(concat_gridWeather_use["station_id"].unique())
concat_gridWeather_use=concat_gridWeather_use[concat_gridWeather_use["station_id"]==stat[0]]
concat_gridWeather_use.head()
concat_gridWeather_use=concat_gridWeather_use.fillna(method="ffill")
concat_gridWeather_use.head()
plt.figure()
plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.legend(["O3"])
plt.xlabel('time')
plt.ylabel('O3(ug/m3)')
plt.plot(range(len(concat_gridWeather_use)),concat_gridWeather_use["temperature"],color='b')

plt.subplot(3,1,2)
plt.legend(["PM10"])
plt.xlabel('time')
plt.ylabel('PM10(ug/m3)')
plt.plot(range(len(concat_gridWeather_use)),concat_gridWeather_use["pressure"],color='g')

plt.subplot(3,1,3)
plt.legend(["PM2.5"])
plt.xlabel('time')
plt.ylabel('PM2.5(ug/m3)')
plt.plot(range(len(concat_gridWeather_use)),concat_gridWeather_use["humidity"],colors='r')


####use concat_gridWeather_use info
def direction_map(df):
    df["direction"]=""
    for i in range(len(df)):
        print(i)
        if df.loc[i,"wind_direction"]>=0 and df.loc[i,"wind_direction"]<45:
            df.loc[i,"direction"]=0
        elif df.loc[i,"wind_direction"]>=45 and df.loc[i,"wind_direction"]<90:
            df.loc[i,"direction"]=1
        elif df.loc[i,"wind_direction"]>=90 and df.loc[i,"wind_direction"]<135:
            df.loc[i,"direction"]=2
        elif df.loc[i,"wind_direction"]>=135 and df.loc[i,"wind_direction"]<180:
            df.loc[i,"direction"]=3
        elif df.loc[i,"wind_direction"]>=180 and df.loc[i,"wind_direction"]<225:
            df.loc[i,"direction"]=4
        elif df.loc[i,"wind_direction"]>=225 and df.loc[i,"wind_direction"]<270:
            df.loc[i,"direction"]=5
        elif df.loc[i,"wind_direction"]>=270 and df.loc[i,"wind_direction"]<315:
            df.loc[i,"direction"]=6           
        elif df.loc[i,"wind_direction"]>=315 and df.loc[i,"wind_direction"]<=360:
            df.loc[i,"direction"]=7
    return df

concat_gridWeather_use=direction_map(concat_gridWeather_use)


pivot_gridWeather_use=pd.crosstab(concat_gridWeather_use.station_id,concat_gridWeather_use.direction)
pivot_gridWeather_use["station_id"]=pivot_gridWeather_use.index
concat_gridWeather_use=pd.merge(concat_gridWeather_use,pivot_gridWeather_use,how = "left",on="station_id")

concat_gridWeather_direction=concat_gridWeather_use[["station_id",0,1,2,3,4,5,6,7]]
concat_gridWeather_direction=concat_gridWeather_direction.drop_duplicates(keep='first')

concat_gridWeather_direction["grid_direction"]=""
for i in range(len(concat_gridWeather_direction)):
    l=list(concat_gridWeather_direction.iloc[i,1:-1])
    m=l.index(max(l))
    concat_gridWeather_direction.iloc[i,-1]=m
concat_gridWeather_direction=concat_gridWeather_direction[["grid_direction","station_id"]]    

concat_gridWeather_use=pd.merge(concat_gridWeather_use,concat_gridWeather_direction,on="station_id",how="left")
concat_gridWeather_use=concat_gridWeather_use.drop(["direction",0,1,2,3,4,5,6,7],axis=1)

### above code get the direction of grid weather sattion

####2.domain feature
def diff_feature(concat_gridWeather_use):
    stations=list(concat_gridWeather_use["station_id"].unique())
    #stations=stations[:1]
    df=pd.DataFrame()
    for station in stations:
        temp_pd=concat_gridWeather_use[concat_gridWeather_use["station_id"]==station]
        temp_pd=temp_pd.sort_values("utc_time")
        t=pd.DataFrame()
        t["humidity_b"]=list(temp_pd["humidity"])[1:]
        t["humidity_f"]=list(temp_pd["humidity"])[:-1]
        t["humidity_diff"]=t["humidity_b"]-t["humidity_f"]
        
        t["pressure_b"]=list(temp_pd["pressure"])[1:]
        t["pressure_f"]=list(temp_pd["pressure"])[:-1]
        t["pressure_diff"]=t["pressure_b"]-t["pressure_f"]
        
        t["temperature_b"]=list(temp_pd["temperature"])[1:]
        t["temperature_f"]=list(temp_pd["temperature"])[:-1]
        t["temperature_diff"]=t["temperature_b"]-t["temperature_f"]
        t=t[["humidity_diff","pressure_diff","temperature_diff"]]
        
        dicx={"humidity_diff":[t.iloc[0,0]],"pressure_diff":[t.iloc[0,1]],"temperature_diff":[t.iloc[0,2]]}
        temp=pd.DataFrame(dicx)  
        temp_x=pd.concat([temp,t],axis=0)  
        temp_x=temp_x.reset_index()
        del temp_x["index"]
        temp_pd["humidity_diff"]=list(temp_x["humidity_diff"])
        temp_pd["pressure_diff"]=list(temp_x["pressure_diff"])
        temp_pd["temperature_diff"]=list(temp_x["temperature_diff"])
        print(max(temp_pd["humidity_diff"]))
        df=pd.concat([df,temp_pd],axis=0)
        
    return df

concat_gridWeather_use=diff_feature(concat_gridWeather_use)

concat_gridWeather_use.to_csv("concat_gridWeather_use.csv")

###3. spatial feature danamic feature
####need to run 5 hours for feature 3

concat_airqQuality_inpl["grid_direction"]=""
concat_airqQuality_inpl["aver_humdiff"]=""
concat_airqQuality_inpl["aver_presdiff"]=""
concat_airqQuality_inpl["aver_tempdiff"]=""
concat_airqQuality_inpl["aver_weather"]=""
concat_airqQuality_inpl["aver_humidity"]=""
concat_airqQuality_inpl["aver_pressure"]=""
concat_airqQuality_inpl["aver_temperature"]=""
concat_airqQuality_inpl["aver_wind_direction"]=""
concat_airqQuality_inpl["aver_wind_speed_x"]=""
concat_airqQuality_inpl["aver_wind_speed_y"]=""


def grid_weather(concat_gridWeather_use,concat_airqQuality_inpl,dic,k):
    def temp_df(station,utc_time,dic):
        df=concat_gridWeather_use[concat_gridWeather_use.station_id.isin(dic[station])]
        df=df[df["utc_time"]==utc_time]
        return df   
    
    for i in range(len(concat_airqQuality_inpl)):
    #for i in range(10):
        if k==1:
            timex=datetime.datetime.strftime(concat_airqQuality_inpl.iloc[i,1],"%Y-%m-%d %H:%M:%S")
            #time=concat_airqQuality_inpl.iloc[i,1].
            temp=temp_df(concat_airqQuality_inpl.iloc[i,0],timex,dic)
        elif k==0:
            temp=temp_df(concat_airqQuality_inpl.iloc[i,0],concat_airqQuality_inpl.iloc[i,1],dic)
        print(i)
        #print(temp)
        try:
            concat_airqQuality_inpl.iloc[i,-11]=pd.DataFrame(temp['grid_direction'].value_counts()).index[0]
        except:
            print("no data")
        concat_airqQuality_inpl.iloc[i,-10]=temp["humidity_diff"].mean()
        concat_airqQuality_inpl.iloc[i,-9]=temp["pressure_diff"].mean()
        concat_airqQuality_inpl.iloc[i,-8]=temp["temperature_diff"].mean()
        try:
            concat_airqQuality_inpl.iloc[i,-7]=pd.DataFrame(temp['weather'].value_counts()).index[0]
        except:
            print("no weather data")
        concat_airqQuality_inpl.iloc[i,-6]=temp["humidity"].mean()
        concat_airqQuality_inpl.iloc[i,-5]=temp["pressure"].mean()
        concat_airqQuality_inpl.iloc[i,-4]=temp["temperature"].mean()
        concat_airqQuality_inpl.iloc[i,-3]=temp["wind_direction"].mean()
        concat_airqQuality_inpl.iloc[i,-2]=temp["speed_x"].mean()
        concat_airqQuality_inpl.iloc[i,-1]=temp["speed_y"].mean()
        
    return concat_airqQuality_inpl
    
concat_airqQuality_inpl_f=grid_weather(concat_gridWeather_use,concat_airqQuality_inpl,dic,0)    

concat_airqQuality_inpl_f.to_csv("concat_airqQuality_feature_done.csv")


#### construct predicted feature 
#test=concat_gridWeather_use[concat_gridWeather_use["station_id"]=="beijing_grid_304"]

####4. statistical feature
statistic_airquality=pd.DataFrame()
statistic_airquality["mean_PM2.5"]=concat_airqQuality_inpl.groupby(["station_id"])["PM2.5"].mean()
statistic_airquality["max_PM2.5"]=concat_airqQuality_inpl.groupby(["station_id"])["PM2.5"].max()
statistic_airquality["min_PM2.5"]=concat_airqQuality_inpl.groupby(["station_id"])["PM2.5"].min()
statistic_airquality["std_PM2.5"]=concat_airqQuality_inpl.groupby(["station_id"])["PM2.5"].std()

statistic_airquality["mean_PM10"]=concat_airqQuality_inpl.groupby(["station_id"])["PM10"].mean()
statistic_airquality["max_PM10"]=concat_airqQuality_inpl.groupby(["station_id"])["PM10"].max()
statistic_airquality["min_PM10"]=concat_airqQuality_inpl.groupby(["station_id"])["PM10"].min()
statistic_airquality["std_PM10"]=concat_airqQuality_inpl.groupby(["station_id"])["PM10"].std()

statistic_airquality["mean_O3"]=concat_airqQuality_inpl.groupby(["station_id"])["O3"].mean()
statistic_airquality["max_O3"]=concat_airqQuality_inpl.groupby(["station_id"])["O3"].max()
statistic_airquality["min_O3"]=concat_airqQuality_inpl.groupby(["station_id"])["O3"].min()
statistic_airquality["std_O3"]=concat_airqQuality_inpl.groupby(["station_id"])["O3"].std()
statistic_airquality["station_id"]=statistic_airquality.index
concat_airqQuality_inpl_f=pd.merge(concat_airqQuality_inpl_f,statistic_airquality,on="station_id")


###5. 站点及时间信息
concat_airqQuality_inpl=concat_airqQuality_inpl_f
concat_airqQuality_inpl["date"]=concat_airqQuality_inpl["utc_time"].apply(lambda x:x.split(' ')[0])
concat_airqQuality_inpl["whatday"]=concat_airqQuality_inpl["date"].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d').strftime("%w"))
concat_airqQuality_inpl["if_week_end"]=concat_airqQuality_inpl["whatday"].apply(lambda x:1 if int(x) in [0,6] else 0)
concat_airqQuality_inpl["if_week_first"]=concat_airqQuality_inpl["whatday"].apply(lambda x:1 if int(x)==1 else 0)
del concat_airqQuality_inpl["whatday"]
del concat_airqQuality_inpl["date"]
del concat_airqQuality_inpl["time_month"]

##对station_type和weather进行编码
concat_airqQuality_inpl=pd.merge(concat_airqQuality_inpl,station_label,on="station_id")
dummies_station=pd.get_dummies(concat_airqQuality_inpl["station_type"])
concat_airqQuality_inpl=pd.concat([concat_airqQuality_inpl,dummies_station],axis=1)
concat_airqQuality_inpl=concat_airqQuality_inpl.drop(['station_type'], axis=1)

dummies_weather=pd.get_dummies(concat_airqQuality_inpl["aver_weather"])
concat_airqQuality_inpl=pd.concat([concat_airqQuality_inpl,dummies_weather],axis=1)
concat_airqQuality_inpl=concat_airqQuality_inpl.drop(['aver_weather'], axis=1)

concat_airqQuality_inpl.to_csv("concat_airqQuality_feature_done.csv")


###feature engineering done

###prepare testing data of 2018-5-1 and 2018-5-2
def get_predict_data(stations):
    predict_df=pd.DataFrame()
    for station in stations:
        timex="2018-05-01 0:0:0"
        timex=datetime.datetime.strptime(timex,"%Y-%m-%d %H:%M:%S")
        for j in range(48):
            #print(j)
            predict_df.loc[len(predict_df),"station_id"]=station
            predict_df.loc[len(predict_df)-1,"utc_time"]=timex+datetime.timedelta(hours = j)
    return predict_df 
predict_df=get_predict_data(stations)

predict_df["grid_direction"]=""
predict_df["aver_humdiff"]=""
predict_df["aver_presdiff"]=""
predict_df["aver_tempdiff"]=""
predict_df["aver_weather"]=""
predict_df["aver_humidity"]=""
predict_df["aver_pressure"]=""
predict_df["aver_temperature"]=""
predict_df["aver_wind_direction"]=""
predict_df["aver_wind_speed_x"]=""
predict_df["aver_wind_speed_y"]=""
    
predict_df=grid_weather(concat_gridWeather_use,predict_df,dic,1)    
predict_df_f=pd.merge(predict_df,statistic_airquality,on="station_id")
predict_df_f=pd.merge(predict_df_f,station_label,on="station_id")
predict_df_f["date"]=predict_df_f["utc_time"].apply(lambda x:str(x).split(' ')[0])
predict_df_f["whatday"]=predict_df_f["date"].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d').strftime("%w"))
predict_df_f["if_week_end"]=predict_df_f["whatday"].apply(lambda x:1 if int(x) in [0,6] else 0)
predict_df_f["if_week_first"]=predict_df_f["whatday"].apply(lambda x:1 if int(x)==1 else 0)
del predict_df_f["whatday"]
del predict_df_f["date"]
dummies_station=pd.get_dummies(predict_df_f["station_type"])
predict_df_f=pd.concat([predict_df_f,dummies_station],axis=1)
predict_df_f=predict_df_f.drop(['station_type'], axis=1)
dummies_weather=pd.get_dummies(predict_df_f["aver_weather"])
predict_df_f=pd.concat([predict_df_f,dummies_weather],axis=1)
predict_df_f=predict_df_f.drop(['aver_weather'], axis=1)

predict_df_f.to_csv("process_data/predictx.csv")


'''
fig,ax=plt.subplots()
x=list(station_label["longitude"])
y=list(station_label["latitude"])
ax.scatter(x,y,c='r')
n=np.arange(35)
for i,txt in enumerate(n):
    ax.annotate(txt,(x[i],y[i]))

batch1=[22,23,24,17]
batch2=[25,21,20,16,19,26]
batch3=[29,28,27,15,14,13,18]
for i in range(len(station_label)):
    if i in batch1:
        station_label.iloc[i,3]="batch1"
    elif i in batch2:
        station_label.iloc[i,3]="batch2"
    elif i in batch3:
        station_label.iloc[i,3]="batch3"
    elif station_label.iloc[i,3]=="Urban Stations":
        station_label.iloc[i,3]="batch4"    
    else:
        station_label.iloc[i,3]="batch5"
        

N = 35
label= ['batch1', 'batch2','batch3','batch4','batch5']
fg = sns.FacetGrid(data=station_label, hue='label', hue_order=label)
fg.map(plt.scatter, 'longitude', 'latitude').add_legend()
plt.title('station location distribution')
plt.show()

'''









        
        
        















