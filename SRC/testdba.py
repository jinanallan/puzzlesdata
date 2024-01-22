# test DTW barycenter averaging for multiple n dim time series

from dtaidistance import dtw_barycenter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#generate random time series
def generate_random_ts(n, l):
    return np.random.randint(1, 10, (n, l)).astype(np.double)

#add an event in the middle of the time series which is a spike of hight 100
def add_event(ts, m):
    ts[0][int(len(ts[0])/2)+m] += 100
    return ts
x = generate_random_ts(1, 200) 
y =  generate_random_ts(1, 200)
z = generate_random_ts(1, 200)
h= generate_random_ts(1, 200)

x = add_event(x,7)
y = add_event(y,8)
z = add_event(z,-4)
h = add_event(h,2)




#make a list of time series
data = [x, y, z,h]
plt.plot(x[0], label='x')
plt.plot(y[0], label='y')
plt.plot(z[0], label='z')
plt.plot(h[0], label='h')

#compute the barycenter
avg=dtw_barycenter.dba_loop(data, use_c=True, c=h)
print(avg)
print(len(avg[0]))
plt.plot(avg[0], label='avg')
plt.legend()
plt.show()