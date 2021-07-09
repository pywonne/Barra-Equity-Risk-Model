'''
@author：Yiwen Pan
因子收益率协方差调整：
Newey-West Adjustment
Eigenfactor Risk  Adjustment
Volatility Regime Adjustment

Portfolio：中证500
'''
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

f = pd.read_excel('C:/Users/panyi/Documents/BarraFactorsLibrary/f_ret_final.xlsx', header=0, index_col=0)
f = f.T # index是日期，columns是因子名称

def var_weighted_NW(F, lambd, delay=2):
    '''
    The process to get the Newey_West Adjusted Covariance Matrix
    '''
    Tn = F.shape[0] # time
    Fn = F.shape[1] # factor
    w = np.array([lambd**n for n in range(Tn)][::-1], dtype='float32')
    w = w/w.sum()
    
    # weighted average of factors
    f_mean_w = np.average(F,axis=0,weights=w)
    
    f_cov_raw = np.array([ F.iloc[:,i].values - f_mean_w[i] for i in range(Fn) ])
    
    # Calculate the cov matrix
    F_raw = np.zeros((Fn,Fn))
    for i in range(Fn):
        for j in range(Fn):
            cov_ij = np.sum( f_cov_raw[i] * f_cov_raw[j] * w ) 
            F_raw[i,j] = cov_ij
    
    cov_nw = np.zeros((Fn,Fn))
    F_NW = 21.*F_raw 
    for d in range(1,delay+1):
        cov_nw_i = np.zeros((Fn,Fn))
        for i in range(Fn):
            for j in range(Fn):
                cov_ij = np.sum( f_cov_raw[i][:-d] * f_cov_raw[j][d:] * w[d:] ) / np.sum(w[d:])
                cov_nw_i[i,j] = cov_ij
        
        F_NW += 21.*( (1-d/(delay+1.)) * (cov_nw_i + cov_nw_i.T) )
    return F_NW

def Newey_West_Adjusted(f,tau=90,length=100,n_start=100,NW=1):
    '''
    f：因子收益率
    tau: 半衰期指数
    length: 计算协方差的长度
    n_start：'n_start-length' to 'n_start' are used to get the covariance
    n_forward: the period ahead to calculate bias statistics B; std of r_ahead/risk_predicted
    '''
    F = f.iloc[n_start-length:n_start,:]
    lambd = 0.5**(1./tau) # 指数权重
    if NW:
        F_NW = var_weighted_NW(F,lambd)
    else:
        F_NW = np.cov(F.T)*21

    D0, U0 = np.linalg.eigh(F_NW)
    Var_eigen = D0
    # r = (f.iloc[n_start:n_start+n_forward,:]+1).cumprod().iloc[-1,:]-1
    
    if not np.allclose(F_NW, U0.dot(np.diag(D0)).dot(U0.T)):
        print('ERROR in eigh')
        return
    
    return F, U0, F_NW, np.sqrt(Var_eigen)

def Eigen_Adjusted(F_NW, U, std_i, length=252, N_mc=1000):
    '''
    EigenFactor Risk Adjustment调整经过NW调整之后的协方差 in order to decrease bias statistics偏差统计量
    N_mc: 模拟次数
    '''
    # Monte Carlo Simulation
    for i in range(N_mc):
        r_mc = np.array([np.random.normal(0, std, length) for std in std_i]) # mean = 0, standard deviation = std
        r_mc = np.dot(U, r_mc)
        F_mc = np.cov(r_mc)
        D_mc, U_mc = np.linalg.eigh(F_mc)
        D = (U_mc.T.dot(F_NW)).dot(U_mc)

        if i == 0:
            v = np.diagonal(D)/D_mc
        else:
            v += np.diagonal(D)/D_mc
    
    v = np.sqrt(v/N_mc)
    return v

def v_fitting(v, a=1.4, n_start_fitting=16):
    '''
    Fitting the simulated bias v(k) using a parabola
    v: 需要修正的
    a: 做一次线性修正经验值取1.4
    n_start_fitting： assign zero weight to the first 15 eigenfactors
    '''
    y = v[n_start_fitting:]
    x = np.array(range(n_start_fitting, f.shape[1]))
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    v_pk = np.array([p(xi) for xi in range(n_start_fitting)] + list(y))
    v_sk = a*(v_pk-1)+1

    return np.array(v_sk)

def Volatility_Adjust(self, cov_Eigen, tau=42):
    '''
    self:原始收益率矩阵，时间长度为252天
    cov_Eigen:经过Eigen调整后的协方差矩阵
    tau：半衰期指数
    '''
    lambd = 0.5**(1/tau)
    w = lambd**np.arange(len(self))[::-1]
    b = self.dot(np.diag(1 / self.std()))
    B_Ft = np.sqrt((b ** 2).sum(axis=1) / self.shape[1])
    lambda_F = np.sqrt(np.sum(B_Ft ** 2 * w)/np.sum(w))
    
    #对角线乘上lambda_F的平方
    cov_VRA = cov_Eigen.values
    dia_i = list(range(len(cov_VRA)))
    cov_VRA[dia_i,dia_i] = lambda_F ** 2 * cov_VRA[dia_i,dia_i]
    
    return pd.DataFrame(cov_VRA, index=f.columns, columns=f.columns)

# length = 252
# n_forward = 21
# tau = 90
# Bias = []
# v_all = []
# for i in range(length, f.shape[0]-n_forward, 21):
#     #---------------Newey-West Adjustment得出调整后协方差矩阵-----------------------------
#     F, U, F_NW_Adjusted, std_i = Newey_West_Adjusted(f, tau=tau,length=length, n_start=i, n_forward=n_forward, NW=1)
#     # print(F_NW_Adjusted)
#     # print(np.all(np.linalg.eigvals(F_NW_Adjusted) >= 0))
   
#     # --------------Eigenfactor Adjustment每日做一次Monte Carlo Simulation---------------
#     vi = Eigen_Adjusted(F_NW_Adjusted, U, std_i, length=252,N_mc=1000)
#     print(vi)
#     v_all.append(vi)
# # Figure 4.1
# # Bias = np.array(Bias)
# # Bias_eigen = [np.std(Bias[:,x]) for x in range(Bias.shape[1])]
# # plt.plot(Bias_eigen,'-*')
# # plt.show()

# # Figure 4.3
# # for i in range(len(v_all)):
# #     plt.plot(v_all[i])
# # plt.show()
# vk = np.array(v_all).mean(axis=0)
# # print('------------------------MEAN BIAS----------------------------------')
# # print(vk)
# adj_vk = v_fitting(vk, a=2, n_start_fitting=16)
# # print('-----------------------ADJUSTED BIAS-------------------------------')
# # print(adj_vk)

# # Bias2 = []
# # for i in range(length,f.shape[0]-n_forward,21):
# #     data_cov, U, F_NW, R_i, Std_i = Newey_West_Adjusted(f,tau=tau,length=length,n_start=i,n_forward=n_forward,NW=1)
# #     s, U = np.linalg.eigh(F_NW)
# #     F_eigen = U.dot(np.diag(adj_vk ** 2).dot(np.diag(s))).dot(U.T)
# #     s2, U2 = np.linalg.eigh(F_eigen)
# #     R_eigen2 = R_i #np.dot(U2.T,R_i)
# #     Bias2.append(R_eigen2/np.sqrt(s2))
# # # Figure 4.4 of UNE4
# # B2=np.array(Bias2).std(axis=0)
# # plt.plot(B2,'-*')
# # plt.show()


# BF_t_all = []
# lambda_F_all=[]
# BF_t_vra_all=[]
# CSV = [] # factor cross-sectional volatility(CSV) on day t
# for i in range(length, f.shape[0]-n_forward, 21):
#     F, U, F_NW, R_i, Std_i = Newey_West_Adjusted(f,tau=tau,n_start=i,length=length
#                                                     ,n_forward=n_forward,NW=1)
#     # ---------------------------Volatility Regime Adjustment-------------------------------
#     D0, U0 = np.linalg.eigh(F_NW_Adjusted)
#     D_hat = np.diag(np.power(adj_vk, 2)).dot(np.diag(D0))
#     F_Eigen_Adjusted = pd.DataFrame(U0.dot(D_hat).dot(U0.T))
  
#     tmp = f.iloc[i-length:i,:]
#     t = f.index.tolist()[i-1] # 天数
#     f_i = Volatility_Adjust(tmp, F_Eigen_Adjusted, tau=42)

#     f_i.to_excel(os.path.join('C:/Users/panyi/Documents/BarraFactorsLibrary/f_adjusted_cov_monthly',t+'.xlsx'))
   


