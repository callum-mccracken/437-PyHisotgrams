#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:43:13 2021

@author: celinemazoukh1
"""

# Import Modules ##
#from __future__ import absolute_import, print_function, division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
import time


rng = 100000

##################
dataName = 'raw-2311-samesp SHORT'
##################

# Load data as a dataframe
print("loading file via pandas")
pdtime = time.time()
df = pd.read_table('{}.txt'.format(dataName),usecols=[6, 7], dtype = np.int64)

pdend=time.time()

print("pd loaded in" , pdend - pdtime)


# work with just the first few elements for now
df = df.head(400)

n = df.shape[0]
dlist = []

'''
for i in range(n):
    jmin = 0
    j = 0
    jmax = n
    jlist=[]
    pdif= 1000000
    while j<jmax:
        diff = df.iloc[i,0] - df.iloc[j,1]
        print("diff (out) ", diff)
        if j==0:
            pdif = diff
        #j += 1
        while abs(diff) <= rng:
            dlist.append(diff)
            print(j)
            jlist.append(j)
            j += 1
            
        if diff>rng and diff>pdif:
            jmax = j
            #break
        
            
        #else:
            #print("loop out")
            #break
        
print(len(jlist))
'''
j = 0
i = 0
pdif = 1000000
while i<n:
    while j<n:
        diff = df.iloc[i,0] - df.iloc[j,1]
        print("diff (out)", diff)
        
        if j==0:
            pdif = diff
        elif j!=0:
            pdif = df.iloc[i,0] - df.iloc[j-1,1]

        
        if abs(diff) <= rng:
            dlist.append(diff)
            print("app", j)
            j += 1
            
        if abs(diff)>rng and abs(diff)>abs(pdif):
            print("leaving")
            i += 1
            
        print("pdif", pdif)
        j += 1

#dpd = pd.DataFrame(dlist)      
#dpd.to_csv('histov2_testOutput.csv')          

plt.figure()
plt.hist(dlist,101)
#plt.plot(xspace, fit_function(xspace, *popt), color='red', linewidth=2.5, label='Fitted function')

# Make the plot nicer.
plt.xlim(-rng,rng)
plt.xlabel('Time Delay (ns)')
plt.ylabel('Number of coincidences')
plt.title('Coincidence Count histogram: {} dataset'.format(dataName))
plt.legend(loc='best')
#plt.savefig('pyHist-J-{n}-dfLength{l}-final.png'.format(n=dataName,l=len(df)),dpi=200)
plt.show()
plt.clf()