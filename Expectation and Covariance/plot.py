import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import pandas as pd
#After having discussion with the professor,
#I'm taking integers for the random discrete uniform distribution.
#The formula used for entropy is summation of pi*log2(pi) which is from Shannon, Father of Communication
N=100
l_M=[]
l_V=[]
l_C=[]
l_E=[]
l_G=[]
for K in np.linspace(100,10000,100):
    K=int(K)
    #N+1 not adding N+1 as I'm adding N
    x1=1+np.random.randint(N, size=K)
    x2=1+np.random.randint(N, size=K)
    x=np.maximum(x1,x2)-x1
    M=np.mean(x)
    V=np.var(x)
    C=np.cov(x,x1)[0,1]
    l_M.append(M)
    l_V.append(V)
    l_C.append(C)
    p_data= pd.value_counts(x)/len(x)
    k=p_data.values
    entropy=sc.entropy(p_data,base=2)
    l_E.append(entropy)
    kl=np.log2(k)
    base2entropy=-np.dot(k,kl)
    l_G.append(base2entropy)
print("Mean,Variance,Covariance,Entropy",np.mean(l_M),np.mean(l_V),np.mean(l_C),np.mean(l_E))

K=np.linspace(100,10000,100)
plt.figure(1)
plt.plot(K,l_M)
plt.xlabel('No of Samples')
plt.ylabel('Mean')
plt.title('Mean of max(x1,x2)-x1')
plt.show()

plt.figure(2)
plt.plot(K,l_V)
plt.xlabel('No of Samples')
plt.ylabel('Variance')
plt.title('Variance of max(x1,x2)-x1')
plt.show()

plt.figure(3)
plt.plot(K,l_C)
plt.xlabel('No of Samples')
plt.ylabel('Covariance(W,X1)')
plt.title('Covariance of max(x1,x2)-x1,x1')
plt.show()

plt.figure(4)
plt.plot(K,l_E)
plt.xlabel('No of Samples')
plt.ylabel('Entropy')
plt.title('Entropy with sc.entropy(p_data,base=2)')
plt.show()


plt.figure(5)
plt.plot(K,l_G)
plt.xlabel('No of Samples')
plt.ylabel('Entropy with respect to log2')
plt.title('Entropy sum of p*log2(p)')
plt.show()

#>>> x = [-2.1, -1,  4.3]
#>>> y = [3,  1.1,  0.12]
#>>> X = np.stack((x, y), axis=0)
#>>> print(np.cov(X))
#[[ 11.71        -4.286     ]
# [ -4.286        2.14413333]]
#>>> print(np.cov(x, y))
#[[ 11.71        -4.286     ]
# [ -4.286        2.14413333]]
#>>> print(np.cov(x))
#11.71
#>>> sc.entropy(p_data)
#2.746993700752302
#>>> 52*0.01*np.log(0.01)
#-2.3946884967138073
#>>> 52*0.01*np.log(0.01) + 0.48*np.log(0.48)
#-2.7469937007523035


