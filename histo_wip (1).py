# -*- coding: utf-8 -*-

'''
HISTOGRAM FOR G(2) COINCIDENCE DATA

[wip]

> first: open raw data file (saved labview code output) and extract the two columns needed

> then: perform the binning and counting
'''

## Import Modules ##
#from __future__ import absolute_import, print_function, division
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import time

#start_time = time.time()

## Define Functions ##



## Playground

rng = 10000 #in ns
width = 200  #in ns
countTime = 20.008 #in seconds
offset = 0
offsetB = 0
resolution = 156e9 #in ps*e9 ie milliseconds
tolerance = 10000


sourceFile = 'raw2-201117-Copy.txt' #'shorter-testData.txt' # file with all the text and info at the top removed; this is just the array of numbers

start_load = time.time()

ChA = (np.loadtxt(sourceFile)[:, 6])*resolution
ChB = (np.loadtxt(sourceFile)[:,7])*resolution

end_load = time.time()

print("load complete in ", end_load-start_load)

#start_time = time.time()

l = len(ChA)

start_A = time.time()
A_matrix = np.array([[Ai]*l for Ai in ChA]).transpose()
end_A = time.time()
print("A matrix created in ", end_A - start_A)
'''
B_matrix = np.array([[Bi]*l for Bi in ChB])
print("B matrix created")
differences = A_matrix - B_matrix
print("differences complete")
countsAB = differences[differences < tolerance].flatten()
print("count list complete")
countsAB = (countsAB + rng) / width

#end_time = time.time()
print(end_time - start_time)

filename = 'trial-ABlist.txt'
#print("writing counts list to file...")
with open(filename, 'w') as countsFile:
    countsFile.writelines(map(str, countsAB))


dat = countsAB #np.loadtxt("trial-ABlist.txt",float)

plt.hist(dat, bins=101)
plt.show()
'''