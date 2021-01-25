# -*- coding: utf-8 -*-

'''
HISTOGRAM FOR G(2) COINCIDENCE DATA

[wip]

> first: open raw data file (saved labview code output) and extract the two columns needed

> then: perform the binning and counting
'''

## Import Modules ##
from __future__ import absolute_import, print_function, division
import numpy as np
import matplotlib.pyplot as plt


## Define Functions ##



## Playground

rng = 10000 #in ns*e6 i.e. milliseconds
width = 200 #e6 #in ns*e6 i.e. milliseconds
countTime = 20.008 #in seconds
offset = 0
offsetB = 0
resolution = 156e9 #in ps*e9 ie milliseconds

'''
sourceFile = 'shorterx2-testData.txt' # file with all the text and info at the top removed; this is just the array of numbers
#dataFile = 'channelCounts.txt'

#with open(sourceFile) as file:

ChA = (np.loadtxt(sourceFile)[:, 6])*0.001
ChB = (np.loadtxt(sourceFile)[:,7])*0.001

#print(ChA[2])
#assert len(ChA) == len(ChB)

countsAB = np.zeros(1)

for i in range(len(ChA)):
    for j in range(len(ChB)):
        diff = np.absolute(ChA[i]-ChB[j])
        print(diff)
        if diff<10000:
            coinc = round((rng+diff)/width)
            np.append(countsAB,coinc)
            
        #while diff < 10000:
            #coinc = round((rng+diff)/width)
            #np.append(countsAB, coinc)
    
    A_ = ChA[i]
    B_ = ChB[i]
    diff =  A_ - B_ - offset
    coinc = (rng+diff)/width
    print(coinc)
    np.append(countsAB,coinc)


    while i < ChA[-1]:
        if abs(diff) < rng:
            coinc = round((rng+diff)/width)
            np.append(countsAB,coinc)
        elif abs(ChA[i+1] - ChB[i] - offset) < rng:
            coincA = round((rng+ChA[i+1] - ChB[i] - offset)/width)
            print(coincA)
            np.append(countsAB, coincA)
        elif abs(ChA[i] - ChB[i+1] - offset) < rng:
            coincB = round((rng+ChA[i] - ChB[i+1] - offset)/width)
            np.append(countsAB, coincB)
        else:
            #print('wack')
            pass

filename = 'trial-ABlist.txt'
print("writing counts list to file...")
with open(filename, 'w') as countsFile:
    countsFile.write(str(countsAB))
    
print(ChA.shape)
print(countsAB.shape)
#print(ChA)
#print(countsAB)
'''

dat = np.loadtxt("trial-ABlist-fromEx.txt",float)

plt.hist(dat, bins=50)
plt.show()
