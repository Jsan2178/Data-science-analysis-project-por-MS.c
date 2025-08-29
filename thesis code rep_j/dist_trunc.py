
#FILE FOR DISTRIBUTIONS

# Uniform 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import uniform 


############## UNIFORM DISTRIBUTION ##################
def dist_uniform(a,b,c):
    dist = uniform(a,b)
    samples = dist.rvs(size=c) 
    return samples

############################ WEIBULL ##########################
from scipy.stats import weibull_min
def weibull_trunc(k, a, b, n, p=0.999, loc=0.0): #n number of samples c is for the compact support so por a,b<0 you need c<0, thus c>a
    #Parameters de la Weibull (k,lam >0):
      # k-> shape  (= 1.14)
      #lam -> scale  (9)
      # truncated limits a, b
    if k <= 0:            raise ValueError("k must be > 0.")
    if not (a < b):       raise ValueError("it needs a < b.")
    if a < loc:           raise ValueError(f"'a' ({a}) < loc ({loc}). adjust 'loc' or range [a,b].")
    lam = (b - a) / (-np.log(1 - p))**(1.0 / k) #this ensures we'll have values for any range with lam scale
    # weibull original
    wb = weibull_min(k, scale=lam, loc=loc)
    
    #CDF on the limits
    Fa, Fb = wb.cdf(a), wb.cdf(b)         # for pdf: f_trunc(x) = f(x)/(F(b)-F(a))
    #Z = Fb - Fa                             # normalization
    
    if not (0.0 <= Fa < Fb <= 1.0):
        raise ValueError(f"Rango without mass: F(a)={Fa:.3g}, F(b)={Fb:.3g}. Check [a,b] y 'loc'.")

    #Generate samples
    U = np.random.rand(n)
    samples = wb.ppf(Fa + U*(Fb-Fa))         
    return samples

#############################  GAMMA ##################################
from scipy.stats import gamma
from scipy.special import gammaincinv
def gamma_trunc(k, lo, hi, n, p=0.99): #n number of samples 
    #Parameters de la Gamma:
      # k-> shape  (= 1.14)
      #alpha -> scale  
      # truncated limits a, b
    if k <= 0:
        raise ValueError("k must be > 0.")
    if not (lo < hi):
        raise ValueError("it needs a < b.")
    alpha = (hi-lo)/gammaincinv(k,p) #Scale for F(b) approx p so we can have negative values also
    # gamma original
    gam = gamma(a=k, scale=alpha, loc=lo)
    
    #CDF on the limits
    Fa, Fb = gam.cdf(lo), gam.cdf(hi)         # for pdf: f_trunc(x) = f(x)/(F(b)-F(a))
    #Z = Fb - Fa                             # normalization
    if Fa >= Fb:
        raise ValueError("The range [a,b] doesn't have mass probability")
    #Generate samples
    U = np.random.rand(n)
    samples = gam.ppf(Fa + U*(Fb-Fa))         
    return samples


########################### GAUSSIAN #####################################################
from scipy.stats import truncnorm
def trunc_norm_sample(a, b, samples, rng=None): #a & b limits
    k=0.2 #standard deviation percent to choose from a given range
    mean, de= (a+b)/2 , k*(b-a)
    a_est, b_est =(a-mean)/de, (b-mean)/de  #fixed range as we want [a,b].
    if rng is None:
        rng= np.random.default_rng() #new generator each time you call this 
    data = truncnorm.rvs(a_est, b_est, loc=mean, scale=de, size=samples)
    return data
