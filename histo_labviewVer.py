#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 00:12:30 2020

@author: celinemazoukh1
"""

# Import Modules ##
#from __future__ import absolute_import, print_function, division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
import time

from scipy.optimize import curve_fit

# Define fit functions.
def fit_function(x, A, mu, sigma,B=0):
    return A * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)) + B

def fit_functionH(x, m,b):
    return m*x + b

rng = 100000
#resol = 156e9
#tolerance = 100000 #floor(rng/resol)
width = 200

##################
dataName = 'raw-2311-fsp'
paper = 'Fine'
##################

# Load data as a dataframe
print("loading file via pandas")
pdtime = time.time()
df = pd.read_table('{}.txt'.format(dataName),usecols=[6, 7], dtype = np.int64)

pdend=time.time()

print("pd loaded in" , pdend - pdtime)

# work with just the first few elements for now
#df = df.head(4000)
#print(df)

# loop through indices:if within range, add to histo and increment both indices
# if not, check whether too positive or too negative and increment accordingly
#dlist = np.zeros(df.shape)
#'''
loopst = time.time()
i = 0
j = 0
n = df.shape[0]
dlist = [] #np.zeros((n,n), dtype=np.int64)
while i < n and j < n:
    diff = df.iloc[i,0] - df.iloc[j,1]
    #print( i, j)
    
    if rng>abs(diff):
        dlist.append(diff)
        
    
    if diff<0:
        i += 1
    else:
        j += 1
    

#dlist = dlist.flatten()
dpd = pd.DataFrame(dlist)      
loopend = time.time()

#dpd = pd.read_csv('raw3_processed_labviewVeroutput.csv',usecols=[1])
#darray = dpd.to_numpy()
#dpd = (dpd+rng)/width

print("loop complete in", loopend-loopst)
#print(len(dlist))

# write the arrays to a txt file
#dpd.to_csv('raw3_processed_labviewVeroutput.csv')

'''
loopst = time.time()
i = 0
j = 0
n = df.shape[0]
dlist = [] #np.zeros((n,n), dtype=np.int64)
while i < n and j < n:
    diff = df.iloc[i,0] - df.iloc[j,1]
    #print( i, j)
    
    if rng>abs(diff):
        dlist.append(diff)
        i += 1
        j += 1
        
    else:
        if diff<0:
            i += 1
        else:
            j += 1
    

#dlist = dlist.flatten()
dpd = pd.DataFrame(dlist)      
loopend = time.time()
#'''
# load and plot the AB counts gen by labview
plt.figure()
coinc = pd.read_table('{}.txt'.format(dataName),usecols=[0, 1], dtype = np.int64)
coinc = coinc.head(101)
'''
# before fitting: slice coinc data to remove bump
wings = coinc.drop('tAB', axis=1) # drops the times column
wings = wings.drop(coinc.index[35:65]) # CHANGE PER EACH SP
wings = wings.to_numpy(dtype= 'float16')
wings=wings.flatten()
print(wings)


# Fit the function to the histogram data.
#xsH = np.linspace(-10000,10000,71)
#poptH, pcovH = curve_fit(fit_functionH, xdata=xsH, ydata=wings)
print(poptH)

'''

coinc.plot(kind='line', x='tAB', y='nAB')
#plt.plot(xsH, fit_functionH(xsH, *poptH), color='red', linestyle='dashed', label='Fitted function')
plt.ylim(0,20000)
plt.xlabel('Time Delay (ns)')
plt.ylabel('Number of Coincidences')
#plt.title('Labview-generated Histogram: {} dataset'.format(dataName))
plt.title('Coincidences Histogram: {} grit sandpaper'.format(paper))

#plt.savefig('labviewHist-{}-final.png'.format(dataName),dpi=200)
plt.savefig('Coincidences Histogram: {} grit sandpaper - ax'.format(paper))
plt.show()

#plt.figure()
#dpd.plot.hist(bins=101)
#plt.show()
'''
print("yeet")

darray = dpd.to_numpy()
# Add histograms of exponential and gaussian data.
data_entries, binscenters, _ =  plt.hist(darray, 101)

# Fit the function to the histogram data.
#popt, pcov = curve_fit(fit_function, xdata=binscenters[1:], ydata=data_entries, p0=[1, 0, 100, 1000])
print(popt)

# Generate enough x values to make the curves look smooth.
xspace = np.linspace(min(binscenters), max(binscenters) , 100000)

# Plot the histogram and the fitted function.
#plt.bar(binscenters, data_entries, width=binscenters[1] - binscenters[0], color='navy', label='Histogram entries')

plt.hist(darray,101)
#plt.plot(xspace, fit_function(xspace, *popt), color='red', linewidth=2.5, label='Fitted function')

# Make the plot nicer.
plt.xlim(min(binscenters),max(binscenters))
#plt.ylim(0,30000)
plt.xlabel('Time Delay (ns)')
plt.ylabel('Number of coincidences')
plt.title('Coincidence Count histogram: {} grit dataset'.format(paper))
plt.legend(loc='best')
#plt.savefig('pyHist-J-{n}-dfLength{l}-FINAL.png'.format(n=dataName,l=len(df)),dpi=200)
plt.show()
plt.clf()
'''