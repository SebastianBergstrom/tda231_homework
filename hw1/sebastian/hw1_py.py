# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 08:53:02 2018

@author: Sebastian
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

def gamma(sigma,alpha,beta):
    return beta**alpha / scipy.special.gamma(alpha) * (sigma**2)**(-alpha-1) * np.exp(-beta/(sigma**2))

def log_factorial(x):
    sum = 0
    for i in range(x):
        sum += np.log(x-i) 
    return sum

def log_gamma(sigma,alpha,beta):
    return alpha*np.log( beta ) - log_factorial(alpha) + (-alpha-1)*np.log( sigma**2 ) - beta/(sigma**2)
    
def sge(X):
    N,p = X.shape
    mu = np.sum(X,axis=0)/N    
    sigma = 1/np.sqrt(N*p) * np.sqrt( np.sum( np.linalg.norm(X-mu,axis=1)**2 ) )
    return mu,sigma    

def post_gamma(sigma,mu,alpha,beta,X):
    n = X.shape[0]
    print(n)
    return log_gamma(sigma, alpha+n , beta + np.sum( np.linalg.norm(X-mu,axis=1)**2 ) / 2 )

def myplot2(X,alpha,beta):
    sigma_start = 0.2
    sigma_end = 125
    mu,_ = sge(X)

    #sigma_vec = np.linspace(sigma_start,sigma_end,1000)
    sigma_vec = np.logspace(-1,2.3,1000)
    gamma_vec = gamma(sigma_vec,alpha,beta)
    post_gamma_vec = np.exp( post_gamma(sigma_vec,mu,alpha,beta,X) )
    
    fig,ax1 = plt.subplots()
    
    color = 'tab:red'
    
    ax1.plot(sigma_vec,gamma_vec,'blue',color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('value')
    ax1.set_ylabel('$\sigma$')
    
    ax2 = ax1.twinx()
    
    color = 'tab:blue'
    
    
    ax2.plot(sigma_vec,post_gamma_vec,'red',color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.show()

    


with open('Documents/GitHub/tda231_homework/hw1/sebastian/dataset0.txt') as dataset:
    data_raw = np.loadtxt(dataset)
    data = data_raw[:,[0,1]]
X=data
mu,_=sge(X)
beta=1
print(beta+ np.sum( np.linalg.norm(X-mu,axis=1)**2 ) / 2)

myplot2(data,10,1)

#%%