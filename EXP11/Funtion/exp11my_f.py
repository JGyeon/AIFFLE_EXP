#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings('ignore') #경고 무시

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# In[54]:


def plot_rolling_statistics(timeseries, window=12):
    
    rolmean = timeseries.rolling(window=window).mean()  # 이동평균 시계열
    rolstd = timeseries.rolling(window=window).std()    # 이동표준편차 시계열

     # 원본시계열, 이동평균, 이동표준편차를 plot으로 시각화해 본다.
    orig = plt.plot(timeseries, color='blue',label='Original')    
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)


# In[55]:


def augmented_dickey_fuller_test(timeseries):
    # statsmodels 패키지에서 제공하는 adfuller 메서드를 호출합니다.
    dftest = adfuller(timeseries, autolag='AIC')  
    
    # adfuller 메서드가 리턴한 결과를 정리하여 출력합니다.
    print('Results of Dickey-Fuller Test:')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    


# In[56]:


def data_ready (name):
    # 데이터 불러오기 
    dataset_filepath = os.getenv('HOME') + '/aiffel/stock_prediction/data/%s.csv'%name
    df = pd.read_csv(dataset_filepath, index_col='Date', parse_dates=True)
    ts = df['Close']
#     print(type(ts))
#     ts.head()

    # 결측치 처리
    ts = ts.interpolate(method='time')
    ts[ts.isna()]  # Time Series에서 결측치가 있는 부분만 Series로 출력합니다. 
    
    # 로그 변환 시도 
    ts_log = np.log(ts)
    
    # 정성적 그래프 분석
    print('정성적 그래프 분석')
    plot_rolling_statistics(ts_log, window=12)


    #정량적 Augmented Dicky-Fuller Test
    print('-'*30)
    print('정량적 Augmented Dicky-Fuller Test')
    augmented_dickey_fuller_test(ts_log)

    #시계열 분해 (Time Series Decomposition)
    decomposition = seasonal_decompose(ts_log, model='multiplicative', period = 30) 

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Residual 안정성 확인
    print('-'*30)
    residual.dropna(inplace=True)
    augmented_dickey_fuller_test(residual)
    
    # 학습, 테스트 데이터셋 생성 
    print('-'*30)
    train_data, test_data = ts_log[:int(len(ts_log)*0.9)], ts_log[int(len(ts_log)*0.9):]
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.plot(ts_log, c='r', label='training dataset')  # train_data를 적용하면 그래프가 끊어져 보이므로 자연스러운 연출을 위해 ts_log를 선택
    plt.plot(test_data, c='b', label='test dataset')
    plt.legend()

    
    # p,q값 구하기 
    plot_acf(ts_log)   # ACF : Autocorrelation 그래프 그리기
    plot_pacf(ts_log)  # PACF : Partial Autocorrelation 그래프 그리기
    plt.show()

    # 1차 차분 구하기    

    print('-'*30)
    print(' 1차 차분 구하기 ')
    diff_1 = ts_log.diff(periods=1).iloc[1:]

    augmented_dickey_fuller_test(diff_1)

    print('-'*30)
    print(' 2차 차분 구하기 ')
        # 2차 차분 구하기
    diff_2 = diff_1.diff(periods=1).iloc[1:]
    
    augmented_dickey_fuller_test(diff_2)
    
    return train_data, test_data


# In[57]:


# train_data, test_data = data_ready('NVDA')


# In[58]:


def result_gap(train_data, test_data, p, q, d):
    # Build Model
    model = ARIMA(train_data, order=(p, q, d))  
    fitted_m = model.fit() 

    print(fitted_m.summary())
    
    #모델 테스트 및 플로팅
    # Forecast : 결과가 fc에 담깁니다. 
    fc = fitted_m.forecast(len(test_data), alpha=0.05)  # 95% conf
    fc = np.array(fc)
    # Make as pandas series
    fc_series = pd.Series(fc, index=test_data.index)   # 예측결과

    # Plot
    plt.figure(figsize=(10,5), dpi=100)
    plt.plot(train_data, label='training')
    plt.plot(test_data, c='b', label='actual price')
    plt.plot(fc_series, c='r',label='predicted price')
    plt.legend()
    plt.show()
    
    # 최종 예측 모델 정확도 측정
    mse = mean_squared_error(np.exp(test_data), np.exp(fc))
    print('MSE: ', mse)

    mae = mean_absolute_error(np.exp(test_data), np.exp(fc))
    print('MAE: ', mae)

    rmse = math.sqrt(mean_squared_error(np.exp(test_data), np.exp(fc)))
    print('RMSE: ', rmse)

    mape = np.mean(np.abs(np.exp(fc) - np.exp(test_data))/np.abs(np.exp(test_data)))
    print('MAPE: {:.2f}%'.format(mape*100))


# In[59]:


# result_gap(train_data, test_data, 0, 2, 1)


# In[ ]:




