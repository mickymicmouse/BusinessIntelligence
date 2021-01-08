# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import pandas as pd


daum=pd.read_csv(r'C:\baterm\daum_keyword.csv', encoding='euc-kr')
kt_population=pd.read_excel(r'C:\baterm\kt_population.xlsx', encoding='euc-kr')
kt_zipcode=pd.read_excel(r'C:\baterm\kt_zipcode.xlsx', encoding='euc-kr')
mango_market=pd.read_excel(r'C:\baterm\mango_market.xlsx', encoding='euc-kr')
mango_user=pd.read_excel(r'C:\baterm\mango_user.xlsx')
bc=pd.read_csv(r'C:\baterm\bc.csv', encoding='euc-kr')
weather=pd.read_csv(r'C:\baterm\weather.csv')


kt_average=pd.read_excel(r'C:\baterm\KT_AVERAGE.xlsx')
kt_average2=pd.read_excel(r'C:\baterm\KT_AVERAGE2.xlsx')
nam_average=pd.read_excel(r'C:\baterm\nam_average.xlsx', encoding='euc-kr')

# 데이터 셋
weather_nam=weather[weather['지점명']=='남산']
weather_nam2=weather_nam[weather['관측일자']>=20190101]

weather_nam3=weather_nam2[weather['관측일자']<=20190630] # 1월 1일 부터 6월 30일 까지
weather_nam3=weather_nam3.rename(columns={'관측일자':'activity_date'})
weather_nam3.to_excel(r'C:\baterm\weather_nam3.xlsx', index=False)
mango_user2=pd.merge(mango_user,weather_nam3)
weather_nam3['activity_date']







address=[]
for i in range(len(mango_market['address'])):
    
    add=mango_market['address'][i].split(' ') 
    address.append(add)
address=pd.DataFrame(address)
address=address.drop(3,axis=1)
address=address.rename(columns={0:'지역(시)',1:'지역(구)',2:'지역(동)'})
mango_market2=pd.concat([mango_market,address],axis=1)
mango_market2=mango_market2.drop('address',axis=1) #시구동으로 분할





bc2=bc.rename(columns={'지역(동)':'AMD_NM'})
bc2=pd.merge(bc2,kt_zipcode[['AMD_NM','AMD_CD']])



p_region=[]
for i in bc2.index:
    sex=bc2['성별'][i]
    age=bc2['연령대'][i]
    if age=='unmatch':
        age='00'
    
    age1=age[0:2]
    colum=sex+' '+age1
    region=bc2['AMD_CD'][i]
    popul=kt_average2[kt_average2['AMD_CD']==region]
    popul=popul.reset_index()
    p_reg=popul[colum][0]
    p_region.append(p_reg)
    
p_region=pd.DataFrame(p_region)
p_region=p_region.rename(columns={0:'지역별 유동인구'})
bc2=pd.concat([bc2, p_region], axis=1)
bc2.to_excel(r'C:\baterm\bc2.xlsx', index=False)
#지역별 나이별 성별 유동인구수 뽑아내서 칼럼 형성



bc2=pd.read_excel(r'C:\baterm\bc2.xlsx')
bc2=pd.merge(bc2, nam_average[['매출년월','기온']])
#매출년월에 따른 날씨


mango_market['latitude']
mango_user['lat']



#EDA--------------------------------------
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
from matplotlib.pyplot import rcParams


rcParams['font.family']='NanumGothic'

font_path = r'C:\WINDOWS\FONTS\NanumGothic.TTF'
fontprop = fm.FontProperties(fname = font_path, size=18)
font_name = fm.FontProperties(fname=font_path).get_name()
rc('font', family= font_name)

num = [len(bc3[bc3['성별']=='남']),len(bc3[bc3['성별']=='여'])]
label = ['남', '여']
plt.bar(label, num, width=0.4)
plt.xlabel('성별', fontproperties=fontprop)
plt.show()

label = ['신용카드','체크카드']
num = [len(bc3[bc3['신용/체크 구분']=='신용']),len(bc3[bc3['신용/체크 구분']=='체크'])]
plt.bar(label, num, width=0.4)
plt.xlabel('신용/체크', fontproperties=fontprop)

plt.show()

label = ['일반한식','스넥','서양음식','편의점']
num = [len(bc3[bc3['업종명']=='일반한식']),len(bc3[bc3['업종명']=='스넥']),len(bc3[bc3['업종명']=='서양음식']),len(bc3[bc3['업종명']=='편 의 점'])]
plt.bar(label, num, width=0.4)
plt.xlabel('업종명', fontproperties=fontprop)
plt.show()


label = ['0대','10대','20대','30대','40대','50대','60대 이상']
num = [len(bc3[bc3['연령대']=='unmatch']),len(bc3[bc3['연령대']=='10대']),len(bc3[bc3['연령대']=='20대']),len(bc3[bc3['연령대']=='30대']),len(bc3[bc3['연령대']=='40대']),len(bc3[bc3['연령대']=='50대']),len(bc3[bc3['연령대']=='60대 이상'])]
plt.bar(label, num, width=0.4)
plt.xlabel('연령대', fontproperties=fontprop)
plt.show()

plt.plot(bc3['금액'])
plt.xlabel('월별 카드 지출 금액')

plt.plot(bc3['기온'])
plt.xlabel('기온')

plt.plot(bc3['지역별 유동인구'])
plt.xlabel('유동인구')
#---------------------------------------------------------





#더미 생성
import pandas as pd
def makedummy(var, number):
    va=var
    file_name=var+'_bin'
    dummy_name='dummy'+var
    var=bc3[var]
    bins=np.linspace(var.min(),var.max(),number)
    bc3[file_name]=np.digitize(var, bins)
    dummy_name=pd.get_dummies(bc3[file_name],prefix=va, drop_first=True)
    return dummy_name

dum_sex=pd.get_dummies(bc2['성별'])
dum_card=pd.get_dummies(bc2['신용/체크 구분'])
dum_store=pd.get_dummies(bc2['업종명'])
dum_age=pd.get_dummies(bc2['연령대'])
dum_storecode=pd.get_dummies(bc2['업종코드'], prefix='bin')

dum_gu=pd.get_dummies(bc['지역(구)'],drop_first=True)
dum_dong=pd.get_dummies(bc['지역(동)'],drop_first=True)

dum_po=makedummy('지역별 유동인구',10)

reg=li()
def vif(lists): #lists는 입력할 값 

    con=[]
    for i in lists:
        X=bc2[np.setdiff1d(lists,i)]
        y=bc2[i]
        
        reg.fit(X,y)
        r2=reg.score(X,y)
        vif=1/(1-r2)
        total=[i,r2,vif]
        con.append(total)
    con=np.array(con)
    con=pd.DataFrame(con)
    con=con.rename(columns={1:'R2', 2:'vif', 0:'feature'})
    return con

lists=['금액', '지역별 유동인구','기온', '건수' ]
vif(lists)


bc2[['금액', '지역별 유동인구','기온', '건수']].corr()


#더미 생성 후 재 결합

bc3=pd.concat((bc2 ,dum_card, dum_store, dum_sex, dum_age),axis=1)




#변수 설정
from sklearn.model_selection import KFold
varlist=bc3.columns

x=bc3[np.setdiff1d(varlist,['건수','금액','AMD_CD','성별', '신용/체크 구분', '업종명', '연령대','매출년월', '지역(구)', '지역(동)','AMD_NM','SGG_NM','unmatch','SGG_CD','SIDO_CD','SIDO_NM','total','m_total','w_total'])]

y=bc3['금액']

# 3번  KFold 지정
cv=KFold(n_splits=5, shuffle=True)

for t,v in cv.split(bc3):
    train_cv=bc3.iloc[t]       # 훈련용
    val_cv=bc3.iloc[v]         # 검증용 분리.
    
    train_X=train_cv[np.setdiff1d(varlist,['건수','금액','성별', '신용/체크 구분', '업종명', '연령대','AMD_CD'
                                           ,'매출년월', '지역(구)', '지역(동)','AMD_NM','SGG_NM','unmatch','SGG_CD','SIDO_CD','SIDO_NM'
                                           ,'total','m_total','w_total'])]    # 훈련용 독립변수들의 데이터,
    train_Y=train_cv.loc[:,'금액']    # 훈련용 종속변수만 있는 데이터
     
    val_X=val_cv[np.setdiff1d(varlist,['건수','금액','성별', '신용/체크 구분', '업종명', '연령대','AMD_CD' ,'매출년월', '지역(구)', '지역(동)','AMD_NM','SGG_NM','unmatch','SGG_CD','SIDO_CD','SIDO_NM','total','m_total','w_total'])]        # 검증용 독립변수들의 데이터,
    val_Y=val_cv.loc[:,'금액']        # 검증용 종속변수만 있는 데이터,


bc4=bc3[bc3['업종명']=='일반한식']
x=bc4[np.setdiff1d(varlist,['건수','AMD_CD','금액','성별', '매출년월','신용/체크 구분','업종명', '연령대','매출년월', '지역(구)', '지역(동)','AMD_NM','SGG_NM','unmatch','SGG_CD','SIDO_CD','SIDO_NM','total','m_total','w_total'])]
y=bc4['금액']

varlist=bc4.columns
for t,v in cv.split(bc4):
    train_cv=bc4.iloc[t]       # 훈련용
    val_cv=bc4.iloc[v]         # 검증용 분리.
    
    train_X=train_cv[np.setdiff1d(varlist,['건수','AMD_CD','금액','성별', '매출년월'
                                           ,'신용/체크 구분','업종명', '연령대','매출년월', '지역(구)'
                                           , '지역(동)','AMD_NM','SGG_NM','unmatch','SGG_CD','SIDO_CD','SIDO_NM'
                                           ,'total','m_total','w_total'])]    # 훈련용 독립변수들의 데이터,
    train_Y=train_cv.loc[:,'금액']    # 훈련용 종속변수만 있는 데이터
     
    val_X=val_cv[np.setdiff1d(varlist,['건수','AMD_CD','금액','성별', '매출년월','신용/체크 구분', '업종명', '연령대','매출년월', '지역(구)', '지역(동)','AMD_NM','SGG_NM','unmatch','SGG_CD','SIDO_CD','SIDO_NM','total','m_total','w_total'])]        # 검증용 독립변수들의 데이터,
    val_Y=val_cv.loc[:,'금액']        # 검증용 종속변수만 있는 데이터





#progress3

#모델 실행
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor as xgb
from xgboost import plot_importance
from xgboost import plot_tree
from xgboost import to_graphviz
from sklearn.ensemble import GradientBoostingRegressor as gb
from sklearn.linear_model import LinearRegression as li
from sklearn.linear_model import Lasso as la
from sklearn.linear_model import Ridge as rid

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import graphviz


#선형회귀 summary----------------------------------------------------

import statsmodels.api as sm
train_X1= sm.add_constant(train_X)
model_li=sm.OLS(train_Y, train_X1)
results=model_li.fit()
results.params
print(results.summary())
#_-------------------------------------------------

# 폰트------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
from matplotlib.pyplot import rcParams


font_path = r'C:/Windows/Fonts/malgunsl.ttf'
fontprop = fm.FontProperties(fname = font_path, size=18)
font_name = fm.FontProperties(fname=font_path).get_name()
rc('font', family= font_name)

# ----------------------------------------------------




#xgboost----------------------------------------------

model =xgb()
model.fit(train_X,train_Y)


plot_importance(model)
plt.show()
scores = cross_val_score(model,x,y,cv=cv1)
plot_importance(model)
plt.rc('font', family=font_path)
plt.show()

r2_score(val_Y, model.predict(val_X))
pred=model.predict(val_X)

#-------------------------------------------------

model2=li()
model2.fit(train_X,train_Y)
model2.score(train_X, train_Y)

mo=model2.coef_

r2_score(val_Y, model2.predict(val_X))

r2=[]
for i in range(-20,20):
    modela=la(alpha=i/10)
    modela.fit(train_X, train_Y)
    a=r2_score(val_Y, modela.predict(val_X))
    r2.append(a)

r2=[]
for i in range(-20,20):
    modela=rid(alpha=i/10)
    modela.fit(train_X, train_Y)
    a=r2_score(val_Y, modela.predict(val_X))
    r2.append(a)

model3=xgb(colsample_bylevel=0.9, colsample_bytree=0.8, gamma=0, max_depth=10, min_child_weight=5, n_estimators=50, n_thread=4)
model3.fit(x,y)
r2_score(val_Y, model.predict(val_X))


cv1=KFold(n_splits=5, shuffle=True)


scores = cross_val_score(model,x,y,cv=cv1)
scores2 = cross_val_score(model2,x,y,cv=cv1)
scores3 = cross_val_score(model3,x,y,cv=cv1)
scores3
scores4 = cross_val_score(model4,x,y,cv=5)
scores


model




# 1번 2번  XGBRegressor() 생성
model5=xgb()

param_grid={'booster' :['gbtree'],
                 'silent':[False],
                 'learning_rate':[0.1,0.2,0.3,0.4,0.5],
                 'max_depth':[3,4,5,6,7,8,10],
                 'min_child_weight':[1],
                 'gamma':[0],
                 'nthread':[4],
            
                 'n_estimators':[50],
                 'objective':['reg:squarederror']}


# 4번  GridSearchCV로 모델 생성
gcv=GridSearchCV(model5, param_grid=param_grid, cv=cv, scoring='r2')


# 5번    
import time
start=time.time()
gcv.fit(x,y)
finish=time.time()
print('passing time', finish-start)
print('final params', gcv.best_params_)
print('best score', gcv.best_score_)
a=gcv.cv_results_
plt.plot(a['mean_test_score'])

# 6번  최적의 파라미터로 xgboost 모델 생성

model6= gcv.best_estimator_
plot_importance(model6)


first_pred=model6.predict(val_X)
r2_score(val_Y, first_pred)
# 7번 
cv = KFold(n_splits=6, shuffle=True)

# 8번
for tidx, vidx in cv.split(train):
    print('predict with KFold')
    train_cv = train.iloc[tidx] 
    val_cv = train.iloc[vidx]
    train_X = train_cv.loc[:,'Pclass':]
    train_Y = train_cv.loc[:,'Survived'] 
    val_X = val_cv.loc[:,'Pclass':]
    val_Y = val_cv.loc[:,'Survived'] 

# 8-1번 각각의 데이터로 xgboost모델로 학습.

import keras 
import tensorflow as tf

tf.keras.models




