# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 01:29:28 2020

@author: TAC
"""

from datetime import datetime
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import numpy as np

import time 
import serial 
  
ser = serial.Serial('COM4', 9600) 
array=[]
times=[]
while True: 
    try:
        value = ser.readline() 
        value=int(value.decode('utf-8')[0])
    except:
        pass
    array.append(value) 
    times.append(datetime.now())
    time.sleep(0.1) 


time1,array1=zip(*[[i,j] for i,j in zip(times,array) if type(j)==int])#remove errored data

 
plt.plot(array1[25:120])
#plt.xticks(range(len(array1[25:120])), time1[25:120],rotation=90)

plt.xticks([])
plt.yticks([])
plt.xlabel('Time')
plt.ylabel('Peak')
plt.savefig('graph.svg',dip=300)


#remove consective duplicated
array2=[]
time2=[]
prev_value = None
for arr,tim in zip(array1,time1): # the : means we're slicing it, making a copy in other words
    if arr != prev_value:
        time2.append(tim)
        array2.append(arr)
        prev_value = arr

#calculare R_R interval
diff=np.diff(time2)
diff=[(i.microseconds)/1000 for i in diff]




plt.plot(array2)
#plt.xticks(range(len(array1[25:120])), time1[25:120],rotation=90)
plt.xticks([])
plt.yticks([])
plt.xlabel('Time')
plt.ylabel('Peak')
plt.savefig('graph1.svg',dip=300)




