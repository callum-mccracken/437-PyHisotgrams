#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:37:09 2020

@author: celinemazoukh1
"""

## Import Modules ##
#from __future__ import absolute_import, print_function, division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
import time

## Define functions ##
def res(x, resolution = 156e9):
    x_ = x*resolution
    return x_

def binpl(x,rng, width, res = 156e9):
    x_ = (rng + (x*res))/width
    return x_

rng = 100000
resol = 156e9
tolerance = 100000 #floor(rng/resol)
width = 200


# Load data as a dataframe
print("loading file via pandas")
pdtime = time.time()
df = pd.read_table('raw2-201117-Copy-copy.txt',usecols=[6, 7], dtype = np.int64)

pdend=time.time()

print("pd loaded in" , pdend - pdtime)

'''
# Load data the old way for comparison purposes
print("loading via old way")
stime = time.time()
sourceFile = 'raw2-201117-Copy-copy.txt'
ChA = (np.loadtxt(sourceFile)[:, 6])
ChB = (np.loadtxt(sourceFile)[:,7])
etime = time.time()

print("load text done in", etime-stime)
'''
# work with just the first few elements for now
df = df.head(10000)



# Loop through the dataframe and check each index of A against that of B until
# difference is out of range
s_search = time.time()
dlist = []
rec = False


for i in tqdm(range(df.shape[0])):
    for j in range(df.shape[0]):
        diff = df.iloc[i,0] - df.iloc[j,1]
        #dlist.append(diff)
        print(diff)
        
        if abs(diff) >= tolerance:
            print("tolerance reached")
            break
        else:
            dlist.append(diff)
            

print("dlist length", len(dlist))
dpd = pd.DataFrame(dlist)        
e_search = time.time()
print("search time ", e_search - s_search)

#print(dpd)

# try plotting histo of the differences

# Build series of coincidences
ABcounts = (rng + dpd)/width


# Plot histogram
plt.figure()
ABcounts.plot.hist(bins=101)
plt.show()
#print(df.iloc[0,0])

plt.figure()
dpd.plot.hist(bins=101)
'''
# Build timebin arrays
N = ((rng/width)*2)+1 # number of bins
print("N = ", N)
timebins = np.linspace(-rng,rng,num=N)
print("timebins size", len(timebins))
'''