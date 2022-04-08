# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 18:33:24 2021

@author: HP
"""

import math


y = [1,1,-1,-1, 1,1,-1,1,-1,-1]
#No of samples
N = 10 #number of sampels
#initilaization  step
d=1.0/N

D = [d]*10
h= []
h.append([1,1,-1,-1,-1,-1,-1,-1,-1,-1])
h.append( [1,1,1,1,1,1,1,1,-1,-1])
h.append ([-1,-1,-1,-1,1,1,-1,1,-1,1])

#number of classfiers
T = 3

#alpha list
alphas =[0,0,0]
for t in range(T):
    #array to check truth of aclassifier  
    I = [0]*N
    
    for i in range(N):
        if h[t][i]!=y[i]:
            I[i]=1
            print(i)
            
    #compute error rate for that classifier
    E = round(sum([i*j for i,j in zip(I,D)]),3)
    print(E)
    
    #assign weight to classifier
    alpha = round(math.log((1-E)/E)/2,3)
    alphas[t]=alpha
    print(alpha)
    
    #update D
    for j in range(N):
        D[j] = D[j] * math.exp(alpha*I[j])

    
    #normalize D so SUM(D)=1
    D_sum = sum(D)
    for j in range(N):
        D[j] = round(D[j] /D_sum,3)
print("weak classfiers alphas :")       
print(alphas)

print("pleaes enter index of sample to classify it:")
x = int(input())

strong_classifier = alphas[0]*h[0][x]+alphas[1]*h[1][x]+alphas[2]*h[2][x]
print(strong_classifier)