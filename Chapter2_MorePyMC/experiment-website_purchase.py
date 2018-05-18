# Test some of the stuff in Chapter 2


# Code to run the A/B test model (whether one website is better than another).
# This example shows how to test whether some observations are likely to have
# been drawn from a particular distribution


#%%  Imports
import pymc3 as pm
import numpy as np
import theano.tensor as tt

from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import scipy.stats as stats

    
#%% First Create simulated data

# Create two datasets, one with larger N than the other

# Asume the probability of purchasing (P_a) is 0.05.
# Then we can simulate purchases with a Bernouli distribution (because with a
# Bernouil, P(X)=1 with probability p_a, or 0 with probability 1-p_a)

p_true = 0.05  # The 'true' probability of purchasing (this is unknown; we're simulating it)
N1 = 1500 # The number of site visits
N2 = 5000

# Generate test data by sampling from bernoulli
occurrences1 = stats.bernoulli.rvs(p_true , size=N1)
occurrences2 = stats.bernoulli.rvs(p_true , size=N2)

#print(occurrences) # Remember: Python treats True == 1, and False == 0
print("In simulation 1 there were {} purchases and {} non-purchases. Observed frequency is: {}".format(\
      np.sum(occurrences1), len(occurrences1[occurrences1 == 0]), np.mean(occurrences1)   ) )

print("In simulation 2 there were {} purchases and {} non-purchases. Observed frequency is: {}".format(\
      np.sum(occurrences2), len(occurrences2[occurrences2 == 0]), np.mean(occurrences2)   ) )
    

#%% Now create models, using the simulated data to estimate our prior (p_a)

with pm.Model() as model1:
    # The probability of someone purchasing from a site (an uninformed prior)
    p = pm.Uniform('p', lower=0, upper=1)
    
    #include the observations, which are Bernoulli
    obs = pm.Bernoulli("obs1", p, observed=occurrences1)
        
    # To be explained in chapter 3
    step = pm.Metropolis()
    trace = pm.sample(18000, step=step)
    burned_trace1 = trace[1000:]
    
with pm.Model() as model2:
    # The probability of someone purchasing from a site (an uninformed prior)
    p = pm.Uniform('p', lower=0, upper=1)
    obs2 = pm.Bernoulli("obs2", p, observed=occurrences2)
    step = pm.Metropolis()
    # To be explained in chapter 3
    step = pm.Metropolis()
    trace2 = pm.sample(18000, step=step)
    burned_trace2 = trace2[1000:]    
    

#%%  Plot the posterior distribution of the unknown prior, p_a
   
figsize(12.5, 4)
plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
plt.vlines(p_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
plt.hist(burned_trace1["p"], bins=50, histtype="stepfilled", normed=True, \
         alpha = 0.5, lw=3, color= 'r', label="N={}".format(N1) )
plt.hist(burned_trace2["p"], bins=50, histtype="stepfilled", normed=True, \
         alpha = 0.5, lw=3, color= 'g', label="N={}".format(N2) )
plt.legend();