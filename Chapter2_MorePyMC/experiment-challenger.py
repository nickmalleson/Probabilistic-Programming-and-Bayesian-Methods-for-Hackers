# Code for the challenger space craft disaster model (Section 2.2.10) (p52)
# Also see paper notes

#%% Init and read data
import pymc3 as pm
import numpy as np
import theano.tensor as tt

from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import scipy.stats as stats


np.set_printoptions(precision=3, suppress=True)
challenger_data = np.genfromtxt("data/challenger_data.csv", skip_header=1,
                                usecols=[1, 2], missing_values="NA",
                                delimiter=",")
#drop the NA values
challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]

# Get the temperatures and defects as separate arrays
temperature = challenger_data[:, 0]
D = challenger_data[:, 1]  # defect or not?

#%%

# Create the model. Probability of failure (p) is logistic. 
# Parameters are represented by normal distributions. (see pg 56)

# Need a logistic function, although this isn't used by PYMC, that seems
# to use tensor flow variables. logistic() is used later for graphs
def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))

with pm.Model() as model:
    beta =  pm.Normal("beta",  mu=0, tau=0.001, testval=0)
    alpha = pm.Normal("alpha", mu=0, tau=0.001, testval=0)
    p = pm.Deterministic("p", 1.0/(1. + tt.exp(beta*temperature + alpha)))
    
    
#%% Connect the model to the observed data
    
# connect the probabilities in `p` with our observations through a
# Bernoulli random variable. (?)
with model:
    observed = pm.Bernoulli("bernoulli_obs", p, observed=D)


#%% 'Run' model
    
# XXXX STILL NOT SURE WHAT THIS ACTUALLY MEANS. WHAT IS HAPPENNING?
# I see that the model is being trained. But how? Is it doing loads of draws
# from the probability distrubtions of the input variables to see what the 
# outcomes are? Then how to get posterior probaility distributions on the
# intput parameters? ... 

with model:
    # Mysterious code to be explained in Chapter 3
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(120000, step=step, start=start, njobs=4) # njobs runs in parallel
    burned_trace = trace[100000::2]
    

#%% Posterior distributions of the parameters
    
# The model was trained in the above chunk. Now loook at the posterior
# distributions of the parameters.
    
alpha_samples = burned_trace["alpha"][:, None]  # best to make them 1d
beta_samples = burned_trace["beta"][:, None]

figsize(12.5, 6)

#histogram of the samples:
plt.subplot(211)
plt.title(r"Posterior distributions of the variables $\alpha, \beta$")
plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\beta$", color="#7A68A6", normed=True)
plt.legend()

plt.subplot(212)
plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\alpha$", color="#A60628", normed=True)
plt.legend();

#%% Expected probabilities

# Calculate the exected probabilities of a deficiency at given temperatures, 
# with 95% credible interval (similar to a confidence interval, but
# not the same, see p60)

t = np.linspace(temperature.min() - 5, temperature.max()+5, 50)[:, None]
p_t = logistic(t.T, beta_samples, alpha_samples)

mean_prob_t = p_t.mean(axis=0)

# PLOT

from scipy.stats.mstats import mquantiles

# vectorized bottom and top 2.5% quantiles for "confidence interval"
qs = mquantiles(p_t, [0.025, 0.975], axis=0)
plt.fill_between(t[:, 0], *qs, alpha=0.7,
                 color="#7A68A6")

plt.plot(t[:, 0], qs[0], label="95% CI", color="#7A68A6", alpha=0.7)

plt.plot(t, mean_prob_t, lw=1, ls="--", color="k",
         label="average posterior \nprobability of defect")

plt.xlim(t.min(), t.max())
plt.ylim(-0.02, 1.02)
plt.legend(loc="lower left")
plt.scatter(temperature, D, color="k", s=50, alpha=0.5)
plt.xlabel("temp, $t$")

plt.ylabel("probability estimate")
plt.title("Posterior probability estimates given temp. $t$");