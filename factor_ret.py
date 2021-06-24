'''
@author: Yiwen Pan
计算各个因子所对应的收益和风险
R_k = w*x*f_k
sigma_k = (w * X * F_k) * X^T * w^T

portfolio:中证500
'''
from higgsboom.MarketData.CSecurityMarketDataUtils import *
from higgsboom.FuncUtils.DateTime import *
secUtils = CSecurityMarketDataUtils('Z:/StockData')
import numpy as np
import pandas as pd
import os
f_ret = pd.read_excel('C:/Users/panyi/Documents/BarraFactorsLibrary/f_ret_final.xlsx', header=0, index_col=0)

# 假设日期为 2021-02-09
def Factor_Return_Risk(t):
    X = pd.read_excel(os.path.join('C:/Users/panyi/Documents/BarraFactorsLibrary/X_daily',t+'.xlsx'), header=0, index_col=0)
    # print(X) # index是股票，columns是因子
    f = f_ret[t] # daily的f
    # print(f) # index是因子，columns是1（日期）
    F = pd.read_excel(os.path.join('C:/Users/panyi/Documents/BarraFactorsLibrary/f_adjusted_cov_monthly',t+'.xlsx'), header=0, index_col=0)
    # print(F) # 因子*因子的协方差矩阵
    indexWeights = secUtils.DailyIndexWeightFrame('000905.SH', t)
    indexWeights.index = indexWeights['StockID']

    for i in indexWeights['StockID']:
        if i not in X.index:
            indexWeights.drop(index=[i], inplace=True)

    # print(indexWeights)
    w = pd.DataFrame(indexWeights['Weight'], index = indexWeights.index)
    w = w.T # 1*N
    w = w.astype('float')
    # print(w)

    R_k = (w.dot(X)) * f
    R_k = R_k.T
    R_k.columns = [t]
    # print(R_k)

    factor_risk = pd.Series(dtype='float64')
    for i in f_ret.index:
        Fk = pd.DataFrame(F[i])
        tmp = w.dot(X).dot(Fk).iloc[0,0]
        tmp2 = X.T.dot(w.T)
        sigma_k = tmp*tmp2.loc[i,:]
        factor_risk[i] = sigma_k['Weight']

    return R_k, factor_risk

PeriodList = TradingDays(startDate='2018-01-01', endDate='2021-04-09')
for i in range(252, len(PeriodList)-21, 21):
    t = PeriodList[i] # 月频
    return_t, risk_t = Factor_Return_Risk(t)
    print('--------------------Factor Return--------------------')
    print(return_t)
    print('---------------------Factor Risk---------------------')
    print(risk_t)