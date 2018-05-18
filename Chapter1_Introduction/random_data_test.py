# -*- coding: utf-8 -*-
"""
Did this with John at coding club. Test whether we get similar results (e.g.
something that looks like a change in number of texts) when drawing from a
random distribution rather than real data. I.e. what are the chances of 
the 'interesting' result occuring with random data.

"""
import pymc3 as pm
import theano.tensor as tt
import scipy.stats as stats

from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt

#%% Real data

figsize(12.5, 3.5)
count_data = np.loadtxt("data/txtdata.csv")
n_count_data = len(count_data)
plt.bar(np.arange(n_count_data), count_data, color="#348ABD")
plt.xlabel("Time (days)")
plt.ylabel("count of text-msgs received")
plt.title("Did the user's texting habits change over time?")
plt.xlim(0, n_count_data);


plt.hist( count_data  , bins = int(max(count_data)) )


plt.hist(np.random.poisson(lam=17, size=len(count_data)), bins = int(max(count_data)))


#%% Run x times on random poisson sample data

# Create 100 random data samples that look a bit like the text message data
test_data = [ np.random.poisson(lam=17, size=len(count_data)) for x in range(100) ]

# Run the model for each of these samples and see how the results vary
for count_data in test_data:
    n_count_data = len(count_data)
    with pm.Model() as model:
        # alpha is the hyperparameter that determines the prior distribution of the
        # two lambda variables (i.e. the one before and after the switch point)
        alpha = 1.0/count_data.mean()
        
        # The two lambda parameters. (We want to see if their posterior distributions change)
        lambda_1 = pm.Exponential("lambda_1", alpha)
        lambda_2 = pm.Exponential("lambda_2", alpha)
        
        # Tau is the point at which the texting behaviour changed. We don't know anything about this 
        # so we assign a uniform prior belief
        tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)
        
        idx = np.arange(n_count_data) # Index
        lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)
        
        observation = pm.Poisson("obs", lambda_, observed=count_data)
        
        step = pm.Metropolis()
        trace = pm.sample(10000, tune=5000,step=step)
        
        lambda_1_samples = trace['lambda_1']
        lambda_2_samples = trace['lambda_2']
        tau_samples = trace['tau']
        figsize(12.5, 10)
        
        #histogram of the samples:
        
        ax = plt.subplot(311)
        ax.set_autoscaley_on(True)
        
        plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
                 label="posterior of $\lambda_1$", color="#A60628", normed=True)
        plt.legend(loc="upper left")
        plt.title(r"""Posterior distributions of the variables
            $\lambda_1,\;\lambda_2,\;\tau$""")
        plt.xlim([15, 30])
        plt.xlabel("$\lambda_1$ value")
        
        ax = plt.subplot(312)
        ax.set_autoscaley_on(True)
        plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
                 label="posterior of $\lambda_2$", color="#7A68A6", normed=True)
        plt.legend(loc="upper left")
        plt.xlim([15, 30])
        plt.xlabel("$\lambda_2$ value")
        
        plt.subplot(313)
        w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
        plt.hist(tau_samples, bins=n_count_data, alpha=1,
                 label=r"posterior of $\tau$",
                 color="#467821", weights=w, rwidth=2.)
        plt.xticks(np.arange(n_count_data))
        
        plt.legend(loc="upper left")
        #plt.ylim([0, .75])
        #plt.xlim([35, len(count_data)-20])
        plt.xlabel(r"$\tau$ (in days)")
        plt.ylabel("probability");
        
        plt.show()

        
        
        


#%%
        
        
        
with open("data/txtdata.csv") as f:
    for line in f:
        print(line)
print("finished")
for line in f:
    print(line)
with f:
    for line in f:
        print(line)    
        