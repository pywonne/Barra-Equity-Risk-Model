'''
@author：Yiwen Pan
计算daily因子收益率向量f和特质收益率向量u

portfolio：中证500
'''
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

# plt.rcParams["font.family"]="STSong"
# plt.rcParams['axes.unicode_minus']=False
LNCAP_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_LNCAP.pkl', 'rb') 
LNCAP_data = pd.DataFrame(pickle.load(LNCAP_file))
Beta_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_Beta.pkl', 'rb') 
Beta_data = pd.DataFrame(pickle.load(Beta_file))
BP_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_BP.pkl', 'rb')
BP_data = pd.DataFrame(pickle.load(BP_file))
Earning_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_Earning.pkl', 'rb')
Earning_data = pd.DataFrame(pickle.load(Earning_file))
Growth_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_Growth.pkl', 'rb')
Growth_data = pd.DataFrame(pickle.load(Growth_file))
Leverage_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_Leverage.pkl', 'rb')
Leverage_data = pd.DataFrame(pickle.load(Leverage_file))
Liquidity_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_Liquidity_sum.pkl', 'rb')
Liquidity_data = pd.DataFrame(pickle.load(Liquidity_file))
Momentum_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_RSTR.pkl', 'rb')
Momentum_data = pd.DataFrame(pickle.load(Momentum_file)) 
NLSize_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_NLSIZE.pkl', 'rb')
NLSize_data = pd.DataFrame(pickle.load(NLSize_file))
Volatility_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_Volatility.pkl', 'rb')
Volatility_data = pd.DataFrame(pickle.load(Volatility_file))

def get_daily_return(t, Index500_Factor):
    '''
    计算股票每日收益率, （当日收盘价/昨日收盘价）-1
    t: 日期
    '''
    RetFrame  = pd.DataFrame(index = Index500_Factor.index, columns =['r'])
    for stock in Index500_Factor.index:
        stock_daily_data = secUtils.StockDailyDataFrame(stock)
        stock_daily_data.index = stock_daily_data['TRADING_DATE']
        if t not in stock_daily_data.index:
            Index500_Factor.drop(index=[stock], inplace=True)
            RetFrame.drop(index=[stock], inplace=True)
            continue
        if(stock_daily_data.at[t, 'TRADE_STATUS'] != '交易'):
            Index500_Factor.drop(index=[stock], inplace=True)
            RetFrame.drop(index=[stock], inplace=True)
            continue        
        today_ret = stock_daily_data.at[t, 'CLOSE']
        yesterday_ret = stock_daily_data.shift(1).at[t, 'CLOSE']
        daily_ret = (today_ret / yesterday_ret) - 1
        RetFrame.at[stock, 'r'] = daily_ret
    return RetFrame, Index500_Factor

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
    temp = x - (x * y).sum() / y.sum()
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

def get_X(t, Index500_StyleFactor, r):
    '''
    构建每日因子暴露矩阵X
    t: 日期
    ''' 
    # 有一些股票没有style factor的数据
    Index500_StyleFactor = Index500_StyleFactor.dropna(how='any')
    for i in r.index:
        if i not in Index500_StyleFactor.index:
            r.drop(index=[i], inplace=True)
    
    # 去极值
    Index500_StyleFactor = Winsorize(Index500_StyleFactor, limit=5)
    # 风格因子标准化（流通市值）
    CAP_data = pd.Series(np.exp(Index500_StyleFactor['LNCAP']))
    CAP_weights = pd.DataFrame(CAP_data / CAP_data.sum()) # 流通市值加权
    CAP_weights.columns =['weight']
    Index500_StyleFactor['LNCAP'] = CAP_standard(Index500_StyleFactor['LNCAP'], CAP_weights['weight'])
    Index500_StyleFactor['Beta'] = CAP_standard(Index500_StyleFactor['Beta'], CAP_weights['weight'])
    Index500_StyleFactor['BP'] = CAP_standard(Index500_StyleFactor['BP'], CAP_weights['weight'])
    Index500_StyleFactor['Earning'] = CAP_standard(Index500_StyleFactor['Earning'], CAP_weights['weight'])
    Index500_StyleFactor['Growth'] = CAP_standard(Index500_StyleFactor['Growth'], CAP_weights['weight'])
    Index500_StyleFactor['Leverage'] = CAP_standard(Index500_StyleFactor['Leverage'], CAP_weights['weight'])
    Index500_StyleFactor['Liquidity'] = CAP_standard(Index500_StyleFactor['Liquidity'], CAP_weights['weight'])
    Index500_StyleFactor['Momentum'] = CAP_standard(Index500_StyleFactor['Momentum'], CAP_weights['weight'])
    Index500_StyleFactor['NLSize'] = CAP_standard(Index500_StyleFactor['NLSize'], CAP_weights['weight'])
    Index500_StyleFactor['Volatility'] = CAP_standard(Index500_StyleFactor['Volatility'], CAP_weights['weight'])
    # 构建因子暴露度矩阵 X（1个国家因子+28个行业因子+10个风格因子）
    # 国家因子暴露
    country_data = pd.Series(np.ones(len(Index500_StyleFactor)), index = Index500_StyleFactor.index, name = 'country').to_frame()
    # 行业因子暴露
    IndusCons = secUtils.DailyIndustryConstituents('SW', 'L1', t)   # 申万 L1, 28个行业分类
    IndusList = []
    for i in Index500_StyleFactor.index:
        for key in IndusCons:
            if i in IndusCons.get(key):
                IndusList.append(key)
                break
    IndusTmp = pd.DataFrame({'行业': IndusList})
    IndusFactor = pd.get_dummies(IndusTmp)
    IndusFactor.index = Index500_StyleFactor.index
    # print(IndusFactor)
    X = pd.concat([country_data, IndusFactor, Index500_StyleFactor], axis = 1)
    # print(X) 
    # X.to_excel(os.path.join('C:/Users/panyi/Documents/BarraFactorsLibrary/X_daily',t+'.xlsx'))
    return X, Index500_StyleFactor, IndusFactor, r


def get_factor_return(t, Index500_StyleFactor):
    '''
    构建每日因子收益率f和特质股票收益率u
    t: 日期
    '''
    r, adj_Style_Factor = get_daily_return(t, Index500_StyleFactor)
    X, adj_Style_Factor, IndusFactor, adj_r = get_X(t,adj_Style_Factor, r)
    
    LNCAP_data = pd.Series(adj_Style_Factor['LNCAP']).to_frame()
    CAP_data = pd.DataFrame(np.exp(LNCAP_data))
    # 构建权重调整矩阵 V
    # 股票特质风险与它的总市值平放根成反比，加权线性回归
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

    f = W.dot(adj_r)
    # 计算特质股票收益率向量u
    u = r - X.dot(f)
    # print(f) #30*1
    return f, u 

def daily_calculate(t):
    Index500_LNCAP = []
    Index500_Beta = []
    Index500_BP = []
    Index500_Earning = []
    Index500_Growth = []
    Index500_Leverage = []
    Index500_Liquidity = []
    Index500_Momentum = []
    Index500_NLSize = []
    Index500_Volatility = []
    IndexCons = secUtils.IndexConstituents('000905.SH', t) # 中证500股票池
    # print(IndexCons)
    for j in LNCAP_data.columns:
        if j in IndexCons:
            Index500_LNCAP.append(LNCAP_data.at[t,j])
            Index500_Beta.append(Beta_data.at[t,j])
            Index500_BP.append(BP_data.at[t,j])
            Index500_Earning.append(Earning_data.at[t,j])
            Index500_Growth.append(Growth_data.at[t,j])
            Index500_Leverage.append(Leverage_data.at[t,j])
            Index500_Liquidity.append(Liquidity_data.at[t,j])
            Index500_Momentum.append(Momentum_data.at[t,j])
            Index500_NLSize.append(NLSize_data.at[t,j])
            Index500_Volatility.append(Volatility_data.at[t,j])
    # print(Index500_LNCAP) # 市值因子
    Index500_dic = {'LNCAP': Index500_LNCAP, 'Beta': Index500_Beta, 'BP': Index500_BP, 'Earning': Index500_Earning, 'Growth': Index500_Growth,
            'Leverage': Index500_Leverage, 'Liquidity': Index500_Liquidity,'Momentum': Index500_Momentum, 'NLSize': Index500_NLSize, 'Volatility': Index500_Volatility} 
    Index500_StyleFactor = pd.DataFrame(Index500_dic, index=IndexCons) # daily style factor
    # print(Index500_StyleFactor)
    # print(Index500_StyleFactor.isnull().any())

    # 正交去除共线性
    Index500_StyleFactor['Liquidity'] = neu_with_intercept(Index500_StyleFactor['LNCAP'], Index500_StyleFactor['Liquidity'])
    Index500_StyleFactor['Volatility'] = neu_with_intercept(Index500_StyleFactor['Beta'], Index500_StyleFactor['Volatility'])
    Index500_StyleFactor['Volatility'] = neu_with_intercept(Index500_StyleFactor['LNCAP'], Index500_StyleFactor['Volatility'])
    
    f, u = get_factor_return(t, Index500_StyleFactor) 

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


PeriodList = TradingDays(startDate='2021-01-01', endDate='2021-04-09')
f_frame = pd.DataFrame() # index行业因子，columns日期
u_frame = pd.DataFrame()
pool = ThreadPool()
f = pool.map(daily_calculate, PeriodList)
f_frame = pd.concat(f, axis = 1)
# u_frame = pd.concat(u, axis = 1)
print(f_frame)
f_frame.to_excel('C:/Users/panyi/Documents/BarraFactorsLibrary/f_ret_final_2021.xlsx')
# u_frame.to_excel('C:/Users/panyi/Documents/BarraFactorsLibrary/u_ret_final.xlsx')

# 风格因子的pure factor return
namelist = ['LNCAP', 'Beta', 'BP', 'Earning', 'Growth', 'Leverage', 'Liquidity', 'Momentum', 'NLSize', 'Volatility']

f_stlye_frame = f_frame.loc[namelist, :]
f_stlye_frame = f_stlye_frame.T
cumf_style_frame = f_stlye_frame.cumsum()
print(cumf_style_frame)
plot_return(cumf_style_frame, namelist)