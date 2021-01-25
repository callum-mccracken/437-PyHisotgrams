#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 23:36:14 2020

@author: celinemazoukh1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define fit function.
def fit_function(x, A, mu, sigma):
    return A * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)) + B

# Add histograms of exponential and gaussian data.
data_entries, binscenters, _ = plt.hist(array, nbins)

# Fit the function to the histogram data.
popt, pcov = curve_fit(fit_function, xdata=binscenters, ydata=data_entries, p0=[1, 0, 100, 1000])
print(popt)

# Generate enough x values to make the curves look smooth.
xspace = np.linspace(0, 6, 100000)

# Plot the histogram and the fitted function.
plt.bar(binscenters, data_entries, width=bins[1] - bins[0], color='navy', label='Histogram entries')
plt.plot(xspace, fit_function(xspace, *popt), color='darkorange', linewidth=2.5, label='Fitted function')

# Make the plot nicer.
plt.xlim(0,6)
plt.xlabel('x axis')
plt.ylabel('Number of entries')
plt.title('Exponential decay with gaussian peak')
plt.legend(loc='best')
plt.show()
plt.clf()
