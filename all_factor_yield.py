'''
@author：Yiwen Pan
计算daily因子收益率向量f和特质收益率向量u

portfolio：全仓
'''
from sys import float_repr_style
from higgsboom.MarketData.CSecurityMarketDataUtils import *
secUtils = CSecurityMarketDataUtils('Z:/StockData')
from higgsboom.FuncUtils.DateTime import *
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import pandas as pd
from scipy import stats
import pickle 
import matplotlib.pyplot as plt
import os


LNCAP_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_LNCAP.pkl', 'rb') 
LNCAP_data = pd.DataFrame(pickle.load(LNCAP_file)).T # index是股票，columns是日期
Beta_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_Beta.pkl', 'rb') 
Beta_data = pd.DataFrame(pickle.load(Beta_file)).T
BP_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_BP.pkl', 'rb')
BP_data = pd.DataFrame(pickle.load(BP_file)).T
Earning_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_Earning.pkl', 'rb')
Earning_data = pd.DataFrame(pickle.load(Earning_file)).T
Growth_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_Growth.pkl', 'rb')
Growth_data = pd.DataFrame(pickle.load(Growth_file)).T
Leverage_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_Leverage.pkl', 'rb')
Leverage_data = pd.DataFrame(pickle.load(Leverage_file)).T
Liquidity_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_Liquidity_sum.pkl', 'rb')
Liquidity_data = pd.DataFrame(pickle.load(Liquidity_file)).T
Momentum_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_RSTR.pkl', 'rb')
Momentum_data = pd.DataFrame(pickle.load(Momentum_file)).T
NLSize_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_NLSIZE.pkl', 'rb')
NLSize_data = pd.DataFrame(pickle.load(NLSize_file)).T
Volatility_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_Volatility.pkl', 'rb')
Volatility_data = pd.DataFrame(pickle.load(Volatility_file)).T

def get_daily_return(t, Style_Factor):
    '''
    计算股票每日收益率, （当日收盘价/昨日收盘价）-1
    t: 日期
    '''
    RetFrame  = pd.DataFrame(index = Style_Factor.index, columns =['r'])
    for stock in Style_Factor.index:
        stock_daily_data = secUtils.StockDailyDataFrame(stock)
        stock_daily_data.index = stock_daily_data['TRADING_DATE']
        if t not in stock_daily_data.index:
            Style_Factor.drop(index=[stock], inplace=True)
            RetFrame.drop(index=[stock], inplace=True)
            continue
        if(stock_daily_data.at[t, 'TRADE_STATUS'] != '交易'):
            Style_Factor.drop(index=[stock], inplace=True)
            RetFrame.drop(index=[stock], inplace=True)
            continue        
        today_ret = stock_daily_data.at[t, 'CLOSE']
        yesterday_ret = stock_daily_data.shift(1).at[t, 'CLOSE']
        daily_ret = (today_ret / yesterday_ret) - 1
        RetFrame.at[stock, 'r'] = daily_ret
    return RetFrame, Style_Factor

def neu_with_intercept(array_x, array_y):
    '''
    正交化以去除共线性
    '''
    # 假定不过原点，有截距项
    y = array_y.replace([np.inf, -np.inf], np.nan).dropna(inplace=False)
    x = array_x.replace([np.inf, -np.inf], np.nan).dropna(inplace=False)
    if x.empty or y.empty:
        print('No data to regress')
        return
    intersectID = list(set(x.index) & set(y.index))
    intersectID.sort()
    x = x[intersectID]
    y = y[intersectID]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    residual = y - x*slope - intercept
    return residual

def CAP_standard(array, floatcap):
    '''
    风格因子通过流通市值标准化
    '''
    x = array.replace([np.inf, -np.inf], np.nan).dropna()
    y = floatcap.replace([np.inf, -np.inf], np.nan).dropna()
    intersectID = list(set(x.index) & set(y.index))
    intersectID.sort()
    x = x[intersectID]
    y = y[intersectID]
    temp = x - (x * y).sum()
    std = x.std()
    return temp / std if std !=0 else temp

def Winsorize(x, limit = 3):
    '''
    因子去极值：均值标准差法，求出每个横截面每个因子的均值和标准差
    '''
    def func(a):
        mean = a.mean()
        standd = a.std()
        a[a < mean-standd*limit] = mean - standd * limit
        a[a > mean+standd*limit] = mean + standd * limit
        return a
    win = x.apply(func)
    return win

def get_X(t, StyleFactor):
    '''
    构建每日因子暴露矩阵X
    t: 日期
    ''' 
    # 构建因子暴露度矩阵 X（1个国家因子+28个行业因子+10个风格因子）
    # 行业因子暴露
    IndusCons = secUtils.DailyIndustryConstituents('SW', 'L1', t)   # 申万 L1, 28个行业分类
    IndusList = []
    stock = []
    for i in StyleFactor.index:
        for key in IndusCons:
            if i in IndusCons.get(key):
                IndusList.append(key)
                stock.append(i)
                break
    IndusTmp = pd.DataFrame({'行业': IndusList})
    IndusFactor = pd.get_dummies(IndusTmp)
    diff = StyleFactor.index.difference(stock).values
    StyleFactor.drop(index=diff, inplace=True)
    IndusFactor.index = StyleFactor.index
    # print(IndusFactor)
    # 风格因子去极值
    StyleFactor = Winsorize(StyleFactor, limit=5)
    # 正交去除共线性
    # 风格因子标准化（流通市值）
    CAP_data = pd.Series(np.exp(StyleFactor['LNCAP']))
    CAP_weights = pd.DataFrame(CAP_data / CAP_data.sum()) # 流通市值加权
    CAP_weights.columns =['weight']
    StyleFactor['LNCAP'] = CAP_standard(StyleFactor['LNCAP'], CAP_weights['weight'])
    StyleFactor['Beta'] = CAP_standard(StyleFactor['Beta'], CAP_weights['weight'])
    StyleFactor['BP'] = CAP_standard(StyleFactor['BP'], CAP_weights['weight'])
    StyleFactor['Earning'] = CAP_standard(StyleFactor['Earning'], CAP_weights['weight'])
    StyleFactor['Growth'] = CAP_standard(StyleFactor['Growth'], CAP_weights['weight'])
    StyleFactor['Leverage'] = CAP_standard(StyleFactor['Leverage'], CAP_weights['weight'])
    StyleFactor['Liquidity'] = neu_with_intercept(StyleFactor['LNCAP'], StyleFactor['Liquidity'])
    StyleFactor['Liquidity'] = CAP_standard(StyleFactor['Liquidity'], CAP_weights['weight'])
    StyleFactor['Momentum'] = CAP_standard(StyleFactor['Momentum'], CAP_weights['weight'])
    StyleFactor['NLSize'] = CAP_standard(StyleFactor['NLSize'], CAP_weights['weight'])
    StyleFactor['Volatility'] = neu_with_intercept(StyleFactor['Beta'], StyleFactor['Volatility'])
    StyleFactor['Volatility'] = neu_with_intercept(StyleFactor['LNCAP'], StyleFactor['Volatility'])
    StyleFactor['Volatility'] = CAP_standard(StyleFactor['Volatility'], CAP_weights['weight'])
    # 国家因子暴露
    country_data = pd.Series(np.ones(len(StyleFactor)), index = StyleFactor.index, name = 'country').to_frame()
    X = pd.concat([country_data, IndusFactor, StyleFactor], axis = 1)
    # print(X) 
    # X.to_excel(os.path.join('C:/Users/panyi/Documents/BarraFactorsLibrary/X_daily',t+'.xlsx'))
    return X, StyleFactor, IndusFactor, diff


def get_factor_return(t, StyleFactor):
    '''
    构建每日因子收益率f和特质股票收益率u
    t: 日期
    '''
    r, adj_Style_Factor = get_daily_return(t, StyleFactor)
    X, adj_Style_Factor, IndusFactor, diff= get_X(t,adj_Style_Factor)
    r.drop(index=diff, inplace=True)

    CAP_data = (np.exp(adj_Style_Factor['LNCAP'])).to_frame()
    # 构建权重调整矩阵 V
    adj_weights = pd.DataFrame(np.sqrt(CAP_data) / np.sqrt(CAP_data).sum())
    adj_weights_series = adj_weights.squeeze() # Convert Dataframe to Series type
    V = pd.DataFrame(np.diag(adj_weights_series), index = adj_Style_Factor.index,  columns= adj_Style_Factor.index)
    # print(V) 

    # 构建约束矩阵 R (K*K-1的约束矩阵)
    # 消除国家因子和行业因子之间共线性，行业市值中性化约束条件：c_I1*f_I1 + c_I2*f_I2 + c_I3*f_I3 + ... + c_Ip*f_Ip = 0
    # c_Ip为该行业所有股票市值之和占全部市值之和的比例
    k = X.shape[1] # 返回X列数，k为因子数量
    diag_R = np.diag(np.ones(k))
    # 移除最后一个行业因子所在的列，并对其所在的行进行权重处理
    diag_R = pd.DataFrame(diag_R, index=X.columns, columns=X.columns)
    R = diag_R.drop(['行业_食品饮料'], axis = 1)
    # c_Ip 行业因子进行权重处理，IndusFactor*CAP_data/CAP_data.sum()
    industry_weights = pd.DataFrame(IndusFactor.T.dot(CAP_data)/CAP_data.sum())
    industry_weights.columns = ['weight']
    adj_industry_weights = pd.DataFrame(-industry_weights.div(industry_weights.iloc[-1]).iloc[:-1])
    for i in adj_industry_weights.index:
        if i != '行业_食品饮料':
            R.loc['行业_食品饮料', i] = adj_industry_weights.at[i, 'weight']
    R = R.values
    # print(R) 30*29

    # 构建权重矩阵W，构建因子收益率向量
    # 截面回归收益率向量为R*（R^T*X^T*V*X*R)^-1 *R^T*X^T*V
    W = R.dot(np.linalg.inv(R.T.dot(X.T).dot(V).dot(X).dot(R))).dot(R.T).dot(X.T).dot(V)
    W = pd.DataFrame(W, index = X.columns, columns = X.index)
    # print(W) 30*500
    # 验算
    # check = W.dot(X).round(4)
    # print(check)

    f = W.dot(r)
    # 计算特质股票收益率向量u
    u = r - X.dot(f)
    # print(f) #30*1
    return f, u 

def daily_calculate(t):
    # print(Index500_LNCAP) # 市值因子
    StyleFactor = pd.concat([LNCAP_data[t], Beta_data[t], BP_data[t], Earning_data[t], Growth_data[t], Leverage_data[t], Liquidity_data[t], Momentum_data[t],
                    NLSize_data[t], Volatility_data[t]], axis=1) # daily style factor
    StyleFactor.columns = namelist
    StyleFactor.dropna(axis=0, how='any', inplace=True)
    # print(StyleFactor)
    f, u = get_factor_return(t, StyleFactor) 

    f.columns = [t]
    u.columns = [t]
    print(f)
    # print(u)
    return f

def plot_return(cumf_ret, namelist):
    colorlist = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'purple', 'orange', 'aqua']
    time = cumf_ret.index.values
    # time = np.array([i.strftime('%Y-%m-%d') for i in time])
    plt.title('Pure Factor Return')
    t = range(0, len(time))
    i = 0
    for name in namelist:
        plt.plot(t, cumf_ret[name], color=colorlist[i], label=name)
        i = i + 1
    plt.legend(loc='lower left', fontsize=10) # 标签位置
    timetic = list(range(0, len(time), len(time)//20))
    plt.xticks(timetic, labels=time[timetic], rotation=90, fontsize=7)
    plt.xlim(0, len(time))

    plt.grid(ls='-.')
    plt.ylabel('Accumulated Return', fontsize=10)

    plt.show()
    return



namelist = ['LNCAP', 'Beta', 'BP', 'Earning', 'Growth', 'Leverage', 'Liquidity', 'Momentum', 'NLSize', 'Volatility']
PeriodList = TradingDays(startDate='2017-06-01', endDate='2017-12-31')
f_frame = pd.DataFrame() # index行业因子，columns日期
u_frame = pd.DataFrame()
pool = ThreadPool()
f = pool.map(daily_calculate, PeriodList)
f_frame = pd.concat(f, axis = 1)
# u_frame = pd.concat(u, axis = 1)
# print(f_frame)
f_frame.to_excel('C:/Users/panyi/Documents/BarraFactorsLibrary/all_stock/f_ret_final_2017_2.xlsx')
# u_frame.to_excel('C:/Users/panyi/Documents/BarraFactorsLibrary/u_ret_final.xlsx')

# 风格因子的pure factor return
# f_stlye_frame = f_frame.loc[namelist, :]
# f_stlye_frame = f_stlye_frame.T
# cumf_style_frame = f_stlye_frame.cumsum()
# print(cumf_style_frame)
# plot_return(cumf_style_frame, namelist)
