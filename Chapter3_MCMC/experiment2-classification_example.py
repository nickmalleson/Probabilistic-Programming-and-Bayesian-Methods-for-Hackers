#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for the Categorical example: 'Unsupervised Clustering using a Mixture Model'
"""

#%% Create data

import pymc3 as pm
import theano.tensor as T
import numpy as np

from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats


figsize(12.5, 4)
data = np.loadtxt("data/mixture_data.csv", delimiter=",")

plt.hist(data, bins=20, color="k", histtype="stepfilled", alpha=0.8)
plt.title("Histogram of the dataset")
plt.ylim([0, None]);
print(data[:10], "...")



#%% Init model with a categorical stochastic variable


with pm.Model() as model:
    p1 = pm.Uniform('p', 0, 1)
    p2 = 1 - p1
    p = T.stack([p1, p2])
    assignment = pm.Categorical("assignment", p, 
                                shape=data.shape,
                                testval=np.random.randint(0, 2, data.shape[0]))
    
print("prior assignment, with p = %.2f:" % p1.tag.test_value)
print(assignment.tag.test_value[:10])

#%% Init Model - include uninformed prior estimates for standard deviations and 
# initial guesses for centres 
with model:
    sds = pm.Uniform("sds", 0, 100, shape=2) # (shape=2 gives two variables)
    centers = pm.Normal("centers", 
                        mu=np.array([120, 190]), 
                        sd=np.array([10, 10]), 
                        shape=2)
    
    center_i = pm.Deterministic('center_i', centers[assignment])
    sd_i = pm.Deterministic('sd_i', sds[assignment])
    
    # and to combine it with the observations:
    observations = pm.Normal("obs", mu=center_i, sd=sd_i, observed=data)
    
print("Random assignments: ", assignment.tag.test_value[:4], "...")
print("Assigned center: ", center_i.tag.test_value[:4], "...")
print("Assigned standard deviation: ", sd_i.tag.test_value[:4])


#%%  'Run' model (sample from it)
# Use Metrolopis to sample the continuous variables and ElementwiseCategorical 
# to sample the categorical.
# Also use pymc to estimate the 'maximum a posterior' (MAP) as a starting location
# to reduce the burn in time

with model:
    step1 = pm.Metropolis(vars=[p, sds, centers])
    step2 = pm.ElemwiseCategorical(vars=[assignment])
    print("Finding MAP")
    start = pm.find_MAP()
    print("Estimated map. Now starting")
    trace = pm.sample(25000, step=[step1, step2], start=start, njobs=4)
    # Also see what happens when we don't start from the estimate of the MAP
    trace2 = pm.sample(25000, step=[step1, step2], njobs=4)
    

#%% Results - visalise posterior distributions usin pymc visualisation library
    
pm.plots.traceplot(trace=trace, varnames=["centers"])

pm.plots.plot_posterior(trace=trace["centers"][:,0])
pm.plots.plot_posterior(trace=trace["centers"][:,1])

# Look at autocorrelation to see how well it converges. The one with better
# starting conditions seems to do slightly better.
pm.plots.autocorrplot(trace=trace, varnames=["centers"]);
pm.plots.autocorrplot(trace=trace2, varnames=["centers"]);


#%% Investigate clusters
# Plot the probability estimate of a point belonging to each cluster

colors = ["#348ABD", "#A60628", "#7A68A6"]
cmap = mpl.colors.LinearSegmentedColormap.from_list("BMH", colors)
assign_trace = trace["assignment"]
plt.scatter(data, 1 - assign_trace.mean(axis=0), cmap=cmap,
        c=assign_trace.mean(axis=0), s=50)
plt.ylim(-0.05, 1.05)
plt.xlim(35, 300)
plt.title("Probability of data point belonging to cluster 0")
plt.ylabel("probability")
plt.xlabel("value of data point");


