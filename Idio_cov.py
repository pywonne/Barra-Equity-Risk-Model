'''
@author：Yiwen Pan
特质股票收益率协方差调整：
Newey-West Adjustment
Structural Model Adjustment
Bayesian Shrinkage Adjustment
Volatility Regime Adjustment

Portfolio：中证500
'''
from higgsboom.MarketData.CSecurityMarketDataUtils import *
secUtils = CSecurityMarketDataUtils('Z:/StockData')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle

LNCAP_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_LNCAP.pkl', 'rb') 
LNCAP_data = pd.DataFrame(pickle.load(LNCAP_file))
u = pd.read_excel('C:/Users/panyi/Documents/BarraFactorsLibrary/u_ret_final.xlsx', header=0, index_col=0)
u = u.sort_index()
u = u.T # index是日期，columns是stock
# print(u)

def var_weighted(self, d=0):
    '''
    计算指数加权方差
    d：滞后期delay的指数加权方差
    return返回指数加权方差 Series
    '''
    weighted_var = pd.DataFrame(columns = self.columns)
    lambd = 0.5**(1./tau) # 指数权重
    if d == 0:
        Tn = len(self)
        w = lambd ** np.arange(Tn)[::-1]
        for i in self.columns:
            stock = self[i]
            weighted_var[i] = [np.sum((stock-stock.mean())**2*w)/np.sum(w)]
        return weighted_var
    else:
        Tn = len(self) - d
        w = lambd ** np.arange(Tn)[::-1]
        for i in self.columns:
            f = self[i].values
            f1 = self[i][:-d].values
            f2 = self[i][d:].values
            weighted_var[i] = [np.sum((f1 - f.mean())*(f2 - f.mean())* w)/np.sum(w)]
        return weighted_var

def Newey_West(self,delay=2,NW=0):
    '''
    进行Newey-West调整，将日频数据转换成月频
    delay：滞后期
    NW：是否进行Newey-West调整
    '''
    var_init = var_weighted(self, d=0)
    if NW:
        for i in range(1, 1+delay):
            var_adjust = var_weighted(self, d=i)
            var_init += var_adjust*(1-i/(delay+1))
    var_TS = 21 * var_init
    return pd.Series(np.array(var_TS)[0], index = var_TS.columns)

def Structural_Adjust(self, t):
    '''
    将特质波动率按照gamma进行结构化调整
    t: 第t日
    '''
    var_TS = Newey_West(self, NW=1) # 进行Newey-West调整
    # 计算gamma_n
    var_e = 1 / 1.35 * (self.quantile(0.75) - self.quantile(0.25)) # 个股n的稳健标准差
    self_adjust = self[(self > -10 * var_e) & (self < 10 * var_e)]
    var_eq = self_adjust.std()
    # print(var_eq)
    Z_e = ((var_eq - var_e)/var_e).abs()
    gamma_n = np.minimum(1, np.maximum(0,(len(self)-60) / 120.0)) * np.minimum(1, np.maximum(0,np.exp(1-Z_e)))
    Stock_nadjust = gamma_n.index[gamma_n == 1] # gamma为1的，大部分股票gamma都为1
    Stock_adjust = gamma_n.index[gamma_n != 1] # gamma 不为1的
    
    X = pd.read_excel(os.path.join('C:/Users/panyi/Documents/BarraFactorsLibrary/X_daily',t+'.xlsx'), header=0, index_col=0)
    # X: index是股票，columns是因子
    # 进行线性回归
    y = np.log(var_TS.loc[Stock_nadjust.intersection(X.index)])
    X_adjust = X.loc[Stock_nadjust.intersection(X.index),:]
    OLS_result = sm.OLS(y, X_adjust).fit()
    coeff = OLS_result.params
    # residue = y.values - X_adjust.dot(coeff).values # 残差
    E_0 = 1 # 用于消除回归残差的指数项影响，约等于1
    var_STR = E_0 * np.exp(X.loc[gamma_n.index,:].dot(coeff))
    # var_STR2 = np.exp(X.loc[gamma_n.index,:].dot(coeff)+residue.mean())
    # print(var_STR2 / var_STR)
    return gamma_n * var_TS + (1 - gamma_n) * var_STR


def Bayesian_Shrinkage(self, t, q=1):
    """
    将特质波动率分为十个组进行Bayesian Shrinkage
    t: 第t日的数据, string
    q：收缩参数，int
    return：返回贝叶斯收缩结果，series
    """
    var_hat = Structural_Adjust(self,t)
    var_hat = pd.DataFrame(var_hat, index = self.columns) # index是股票， columns是波动率
    var_hat['Code'] = var_hat.index
    var_hat.columns = ['Variance','Code'] 
    # 按照流通市值大小排序
    CAP_Series = pd.Series(index = var_hat.index, dtype='float64')
    for i in var_hat.index:
        CAP_Series[i] = np.exp(LNCAP_data.at[t,i])
    var_hat['Capital'] = CAP_Series
    var_hat['Capital'] = var_hat['Capital'] / var_hat['Capital'].sum() # 市值权重
    var_hat.sort_values(by='Capital', inplace=True)
    var_hat.reset_index(drop=True, inplace=True)
    group_num = len(var_hat) / 10 # 十分组，每组个数
    res = pd.DataFrame()
    for i in range(10):
        vh_temp = var_hat.loc[(var_hat.index < (i+1)*group_num) * (var_hat.index >= i*group_num)]
        var_mean = (vh_temp['Variance'] * vh_temp['Capital']).sum() # 某一组的波动率均值为市值加权均值
        delta_Sn = np.sqrt(((vh_temp['Variance'] - var_mean) ** 2).mean())
        vn = 1 / (delta_Sn / (q * np.abs(vh_temp['Variance'] - var_mean)) + 1)
        vh_temp['Variance'] = vn * var_mean + (1 - vn) * vh_temp['Variance']
        res = pd.concat([res, vh_temp[['Code','Variance']]])
    result = pd.Series(res['Variance'])
    result.index = res['Code'].values
    return result

def Volatility_Adjust(self, t, tau=42):
    '''
    各个截面上的股票之间有一定相互的影响，进行Volatility的调整
    self: 特质收益率
    t: 第t天
    tau = 指数权重半衰期
    '''
    lambd = 0.5**(1./tau)
    w = lambd ** np.arange(len(self))[::-1]
    b =  self.dot(np.diag(1 / self.std()))  # std为一个每支股票在一段时间内的std的Series
    BS_t = np.sqrt((b ** 2).sum(axis=1) / self.shape[1]) # 等权平均
    lambda_F = np.sqrt(np.sum(BS_t ** 2 * w) / np.sum(w))
    var_BSA = Bayesian_Shrinkage(self, t)
    var_VRA = lambda_F ** 2 * var_BSA
    var_VRA = var_VRA.sort_index()
    var_VRA = var_VRA.to_frame()
    var_VRA.columns = [t]
    return var_VRA

'''
时间窗口取252天
'''
length = 252 
n_forward = 21
tau = 90
frequency = 21
u_cov = pd.DataFrame()
for i in range(length, u.shape[0], frequency):
    tmp = u.iloc[i-length:i, :]
    tmp = tmp.dropna(axis = 1)
    t = u.index.tolist()[i-1] # 天数
    u_i = Volatility_Adjust(tmp, t) # 特质协方差delta为一个N*N的diagonal matrix
    u_i = u_i.sort_index()
    u_i = u_i.squeeze()
    u_i = pd.DataFrame(np.diag(u_i), index = tmp.columns, columns = tmp.columns)
    print(t)
    print(u_i)
    # u_i.to_excel(os.path.join('C:/Users/panyi/Documents/BarraFactorsLibrary/u_adjusted_cov_monthly',t+'.xlsx'))
    

