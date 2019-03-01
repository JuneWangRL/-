#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 18:35:42 2018

@author: junewang
"""
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import lightgbm as lgb
import datetime

os.environ['KMP_DUPLICATE_LIB_OK']='True'

station_label=pd.read_csv("/Users/junewang/Desktop/5002 Data Mining/msbd5002project_60/src/process_data/station_label.csv",header=None)
station_label.columns = ['station_id', 'longitude', 'latitude','station_type']

concat_airqQuality_train=pd.read_csv("/Users/junewang/Desktop/5002 Data Mining/msbd5002project_60/src/process_data/concat_airqQuality_feature_done.csv")
concat_airqQuality_train=concat_airqQuality_train.iloc[:,1:]
concat_airqQuality_train["CO1"]=concat_airqQuality_train["CO"]
del concat_airqQuality_train["CO"]
concat_airqQuality_train["SO21"]=concat_airqQuality_train["SO2"]
del concat_airqQuality_train["SO2"]
concat_airqQuality_train["NO21"]=concat_airqQuality_train["NO2"]
del concat_airqQuality_train["NO2"]
concat_airqQuality_train["WIND1"]=concat_airqQuality_train["WIND"]
del concat_airqQuality_train["WIND"]
concat_airqQuality_train["SNOW1"]=concat_airqQuality_train["SNOW"]
del concat_airqQuality_train["SNOW"]
concat_airqQuality_train.rename(columns={ concat_airqQuality_train.columns[-1]: "SNOW",
                                         concat_airqQuality_train.columns[-2]: "WIND",
                                         concat_airqQuality_train.columns[-3]: "NO2",
                                         concat_airqQuality_train.columns[-4]: "SO2",
                                         concat_airqQuality_train.columns[-5]: "CO",
                                         }, inplace=True)

pre_df=pd.read_csv("/Users/junewang/Desktop/5002 Data Mining/msbd5002project_60/src/process_data/pre_df_done.csv")
pre_df=pre_df.iloc[:,1:]
pre_df["WIND"]=0
pre_df["SNOW"]=0
def interpolate(df):
    stations=(df["station_id"].unique())
    res_df=pd.DataFrame()
    for station in stations:
        dfx=df[df["station_id"]==station]
        dfx=dfx.interpolate(method="cubic")
        res_df=res_df.append(dfx)
    return res_df
concat_airqQuality_train=interpolate(concat_airqQuality_train)
concat_airqQuality_train=concat_airqQuality_train.drop(["CO","SO2","NO2"],axis=1)
pre_df=pre_df.drop(["CO","SO2","NO2"],axis=1)
batch1=[22,23,24,17]
batch2=[25,21,20,16,19,26]
batch3=[29,28,27,15,14,13,18]
station_label["batch"]=""
for i in range(len(station_label)):
    if i in batch1:
        station_label.iloc[i,4]="batch1"
    elif i in batch2:
        station_label.iloc[i,4]="batch2"
    elif i in batch3:
        station_label.iloc[i,4]="batch3"
    elif station_label.iloc[i,3]=="Urban Stations":
        station_label.iloc[i,4]="batch4"    
    else:
        station_label.iloc[i,4]="batch5"
        
stations=list(concat_airqQuality_train["station_id"].unique())
batch1=list(station_label[station_label["batch"]=='batch1']["station_id"])
batch2=list(station_label[station_label["batch"]=='batch2']["station_id"])
batch3=list(station_label[station_label["batch"]=='batch3']["station_id"])
batch4=list(station_label[station_label["batch"]=='batch4']["station_id"])
batch5=list(station_label[station_label["batch"]=='batch5']["station_id"])


def get_train_set(concat_airqQuality_train,stations):
    train_Traffic=concat_airqQuality_train[concat_airqQuality_train.station_id.isin(stations)]
    string_time="2018-4-30 23:0:0"
    time = datetime.datetime.strptime(string_time, '%Y-%m-%d %H:%M:%S')
    #time_eval_split = time - datetime.timedelta(days = 32)
    time_test_split = time - datetime.timedelta(days = 2)

    train_Traffic["utc_time"]=train_Traffic["utc_time"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
    train_Traffic_trian=train_Traffic[train_Traffic["utc_time"]<time_test_split]
    #train_Traffic_eval=train_Traffic[(train_Traffic["utc_time"]>=time_eval_split) & (train_Traffic["utc_time"]<=time_test_split)]
    train_Traffic_test=train_Traffic[train_Traffic["utc_time"]>=time_test_split]
    #train_Traffic_trian=train_Traffic_trian.reset_index()
    #train_Traffic_eval=train_Traffic_eval.reset_index()
    #train_Traffic_test=train_Traffic_test.reset_index()
    #del train_Traffic_trian['index']
    #del train_Traffic_eval['index']
    #del train_Traffic_test['index']
    #print(train_Traffic_trian.columns)
    train_y=train_Traffic_trian[["PM10","PM2.5","O3"]]
    
    train_x=train_Traffic_trian.drop(["PM10","PM2.5","O3",'utc_time','station_id'],axis=1)
    
    test_y=train_Traffic_test[["PM10","PM2.5","O3"]]
    test_x=train_Traffic_test.drop(["PM10","PM2.5","O3",'utc_time','station_id'],axis=1)
    return train_Traffic_test,train_x,train_y,test_x,test_y


def Attri(train_Traffic_test,attri,train_y,test_y):
    train_Traffic_label=pd.DataFrame(train_Traffic_test["station_id"])
    train_Traffic_label["real_y"]=test_y[attri]
    train_y=pd.DataFrame(train_y[attri])
    test_y=pd.DataFrame(test_y[attri])
    return test_y,train_y,train_Traffic_label

class lgbm_model:
    def __init__(self):
        self.model =lgb.LGBMRegressor(num_leaves=8,
                        learning_rate=0.001,
                        n_estimators=10000,
                        max_bin=15)
        
        self.pre_res=pd.DataFrame()
        self.train_res=pd.DataFrame()
    def train(self, train_x, train_y, test_x):
        # 5-folds cross-valiadation
        trainx,evalx,trainy,evaly=train_test_split(train_x,train_y,test_size=0.2,random_state=9)

        lgb_model = self.model.fit(trainx, trainy,
                            eval_set=[(evalx, evaly)],
                            eval_metric='l1',
                            early_stopping_rounds=5)
        print("----------")
        test_pred = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration_)
        train_pred = lgb_model.predict(train_x, num_iteration=lgb_model.best_iteration_)
        #train_pred=np.expm1(train_pred)
        #accuracy_lgb_train=mean_squared_error(train_y, train_pred)
        #accuracy_lgb_test=mean_squared_error(test_y, test_pred)
        #return  train_x,eval_x,train_y,eval_y
        #print("training:",accuracy_lgb_train)
        #print("testing:",accuracy_lgb_test)
        return test_pred,train_pred

def smape(actual, predicted):
    dividend= np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)
    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator!=0, casting='unsafe'))
      
lgb_model=lgbm_model()
AQI="O3"
batch=batch1
train_Traffic_test,train_x,train_y,test_x,test_y=get_train_set(concat_airqQuality_train,batch)
test_y,train_y,train_Traffic_label=Attri(train_Traffic_test,AQI,train_y,test_y)
test_pred,train_pred=lgb_model.train(train_x,train_y,test_x)
train_Traffic_label["pre_y"]=test_pred
res=smape(train_Traffic_label["real_y"], test_pred)
print("testing error:",mean_squared_error(train_Traffic_label["real_y"], train_Traffic_label["pre_y"]))
print("training error:",mean_squared_error(train_y, train_pred))
print(res)


####predicting final result.
station_l=station_label[["station_id","batch"]]
pre_df_1=pre_df.merge(station_l,on="station_id")

pre_batch1=pre_df_1[pre_df_1["batch"]=="batch1"]
pre_batch2=pre_df_1[pre_df_1["batch"]=="batch2"]
pre_batch3=pre_df_1[pre_df_1["batch"]=="batch3"]
pre_batch4=pre_df_1[pre_df_1["batch"]=="batch4"]
pre_batch5=pre_df_1[pre_df_1["batch"]=="batch5"]

submit_batch1=pre_batch1[["station_id","utc_time"]]
submit_batch2=pre_batch2[["station_id","utc_time"]]
submit_batch3=pre_batch3[["station_id","utc_time"]]
submit_batch4=pre_batch4[["station_id","utc_time"]]
submit_batch5=pre_batch5[["station_id","utc_time"]]

## preparing training data.
pre_batch1=pre_batch1.drop(['batch',"utc_time","station_id"], axis=1)
pre_batch2=pre_batch2.drop(['batch',"utc_time","station_id"], axis=1)
pre_batch3=pre_batch3.drop(['batch',"utc_time","station_id"], axis=1)
pre_batch4=pre_batch4.drop(['batch',"utc_time","station_id"], axis=1)
pre_batch5=pre_batch5.drop(['batch',"utc_time","station_id"], axis=1)


def get_train(concat_airqQuality_train,batch):
    temp=concat_airqQuality_train[concat_airqQuality_train.station_id.isin(batch)]
    train_y=temp[["PM10","PM2.5","O3"]]
    
    train_x=temp.drop(["PM10","PM2.5","O3",'utc_time','station_id'],axis=1)
    return train_x,train_y

train_x,train_y=get_train(concat_airqQuality_train,batch1) 
final_pred1,train_pred=lgb_model.train(train_x,train_y["O3"],pre_batch1)
final_pred2,train_pred=lgb_model.train(train_x,train_y["PM2.5"],pre_batch1)
final_pred3,train_pred=lgb_model.train(train_x,train_y["PM10"],pre_batch1)

submit_batch1["O3"]=final_pred1
submit_batch1["PM2.5"]=final_pred2
submit_batch1["PM10"]=final_pred3
train_x,train_y=get_train(concat_airqQuality_train,batch1) 
final_pred1,train_pred=lgb_model.train(train_x,train_y["O3"],pre_batch1)
final_pred2,train_pred=lgb_model.train(train_x,train_y["PM2.5"],pre_batch1)
final_pred3,train_pred=lgb_model.train(train_x,train_y["PM10"],pre_batch1)

submit_batch1["O3"]=final_pred1
submit_batch1["PM2.5"]=final_pred2
submit_batch1["PM10"]=final_pred3

train_x,train_y=get_train(concat_airqQuality_train,batch2) 
final_pred1,train_pred=lgb_model.train(train_x,train_y["O3"],pre_batch2)
final_pred2,train_pred=lgb_model.train(train_x,train_y["PM2.5"],pre_batch2)
final_pred3,train_pred=lgb_model.train(train_x,train_y["PM10"],pre_batch2)

submit_batch2["O3"]=final_pred1
submit_batch2["PM2.5"]=final_pred2
submit_batch2["PM10"]=final_pred3

train_x,train_y=get_train(concat_airqQuality_train,batch3) 
final_pred1,train_pred=lgb_model.train(train_x,train_y["O3"],pre_batch3)
final_pred2,train_pred=lgb_model.train(train_x,train_y["PM2.5"],pre_batch3)
final_pred3,train_pred=lgb_model.train(train_x,train_y["PM10"],pre_batch3)

submit_batch3["O3"]=final_pred1
submit_batch3["PM2.5"]=final_pred2
submit_batch3["PM10"]=final_pred3

train_x,train_y=get_train(concat_airqQuality_train,batch4) 
final_pred1,train_pred=lgb_model.train(train_x,train_y["O3"],pre_batch4)
final_pred2,train_pred=lgb_model.train(train_x,train_y["PM2.5"],pre_batch4)
final_pred3,train_pred=lgb_model.train(train_x,train_y["PM10"],pre_batch4)

submit_batch4["O3"]=final_pred1
submit_batch4["PM2.5"]=final_pred2
submit_batch4["PM10"]=final_pred3

train_x,train_y=get_train(concat_airqQuality_train,batch5) 
final_pred1,train_pred=lgb_model.train(train_x,train_y["O3"],pre_batch5)
final_pred2,train_pred=lgb_model.train(train_x,train_y["PM2.5"],pre_batch5)
final_pred3,train_pred=lgb_model.train(train_x,train_y["PM10"],pre_batch5)

submit_batch5["O3"]=final_pred1
submit_batch5["PM2.5"]=final_pred2
submit_batch5["PM10"]=final_pred3

submission=pd.concat([submit_batch1,submit_batch2,submit_batch3,submit_batch4,submit_batch5],axis=0)
submission=submission.reset_index()
submission=submission.sort_values(["station_id","utc_time"])
del submission["index"]
'''
submission["day"]=submission["utc_time"].apply(lambda x: int(x[8:10]))
submission["hour"]=submission["utc_time"].apply(lambda x: int(x.split(' ')[1][:2]))

result_submission = submission[[
    'station_id', 'day', 'hour', 'PM2.5', 'PM10', 'O3']]
station_dic = {}
station_dic_reverse = {}
station_sbumission_list = ['dongsi_aq', 'tiantan_aq', 'guanyuan_aq', 'wanshouxigong_aq', 'aotizhongxin_aq', 'nongzhanguan_aq', 'wanliu_aq', 'beibuxinqu_aq',
                           'zhiwuyuan_aq', 'fengtaihuayuan_aq', 'yungang_aq', 'gucheng_aq', 'fangshan_aq', 'daxing_aq', 'yizhuang_aq', 'tongzhou_aq',
                           'shunyi_aq', 'pingchang_aq', 'mentougou_aq', 'pinggu_aq', 'huairou_aq', 'miyun_aq', 'yanqin_aq', 'dingling_aq', 'badaling_aq',
                           'miyunshuiku_aq', 'donggaocun_aq', 'yongledian_aq', 'yufa_aq', 'liulihe_aq', 'qianmen_aq', 'yongdingmennei_aq', 'xizhimenbei_aq',
                           'nansanhuan_aq', 'dongsihuan_aq']
for i in range(len(station_sbumission_list)):
    station_dic[station_sbumission_list[i]] = i
    station_dic_reverse[i] = station_sbumission_list[i]
result_submission['station_id'] = result_submission['station_id'].apply(
    lambda x: station_dic[x])


def rep_hour(day, hour):
    if day > 1:
        hour += 24
    return hour


result_submission['hour'] = result_submission.apply(
    lambda row: rep_hour(row['day'], row['hour']), axis=1)
result_submission['hour'] = result_submission['hour'].astype(int)
final_res_sub = result_submission.sort_values(by=['station_id', 'hour'])
final_res_sub['station_id'] = final_res_sub['station_id'].apply(
    lambda x: station_dic_reverse[x])


def rep_id(station_id, hour):
    hour = str(int(hour))
    res = station_id + '#' + hour
    return res


final_res_sub['station_id'] = final_res_sub.apply(
    lambda row: rep_id(row['station_id'], row['hour']), axis=1)
final_res_sub = final_res_sub.reset_index()
final_res_sub.drop(['index', 'day', 'hour'], axis=1, inplace=True)
final_res_sub.columns = ['test_id', 'PM2.5', 'PM10', 'O3']
final_res_sub.to_csv('./submissiontest.csv', index=None)
'''


num=[]
for i in range(48):
    num.append(i)
num_all=[]
for i in range(35):
    num_all.extend(num)

submission["num"]=num_all 
submission["test_id"]=""
for i in range(len(submission)):
    submission.iloc[i,-1]=submission.iloc[i,0]+"#"+str(submission.iloc[i,5])

submission=submission[["test_id","PM2.5","PM10","O3"]]

ans=pd.read_csv("/Users/junewang/Desktop/ans.csv")
ans=ans.fillna(method="ffill")
def smape(actual, predicted):
    dividend= np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)
    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator!=0, casting='unsafe'))


res1=smape(ans["PM25_Concentration"], submission["PM2.5"])
res2=smape(ans["PM10_Concentration"], submission["PM10"])
res3=smape(ans["O3_Concentration"], submission["O3"])
print(res1+res2+res3)




### plot tendency 
'''
def plot(train_Traffic_label):
    sta=list(train_Traffic_label["station_id"].unique())
    for s in sta:
        temp=train_Traffic_label[train_Traffic_label["station_id"]==s]
        plt.plot(range(len(temp)),temp["pre_y"],color="r")
        plt.plot(range(len(temp)),temp["real_y"])
        plt.title(s)
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('2.5(ug/m3)')
        plt.show()

plot(train_Traffic_label)

fig=plt.figure(figsize=(8,8))
s=[]
for i in range(35):
    plt.subplot(6,6,i+1)
    temp=submission.iloc[i*48:(i+1)*48-1,:]
    plt.plot(range(len(temp)),temp[attri])
    plt.xticks([])
    plt.yticks([])
    s.append(temp.iloc[0,0][:-2])
fig.suptitle(attri+" Tendency",fontsize=16,x=0.53,y=0.95)

'''

