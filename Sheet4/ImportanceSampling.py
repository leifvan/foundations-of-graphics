import numpy as np
import matplotlib.pyplot as plt
from math import exp, sqrt, pi


# f: target function, which we want to integrate
# ...


# p_n: normal distribution for samples
sigma2 = 0.3
mu = 1.2
# ...

# p_p: polynomial fit of the target function
# ...
# (make sure, the fitting is computed only once and not on every invocation of p_p !

# plot the function graphs:
plt.figure("Functions")
# ...


"""
Uses rejection sampling to generate samples
n: amount of samples to generate
d: distribution to draw samples from
max: maximum y value that is used to generate the samples
returns: x and y values of the samples in the shape (n, 2)
"""
def GenSamples(n, d, max):
	s = np.zeros((n, 2))
	# ...
	return s


# Plot results of GenSamples()
# Hint: 0.8 is a reasonable value for the max parameter of GenSamples
plt.figure("Normal Distribution")
# ...

plt.figure("Polynomial")
# ...


"""
p: the function to integrate
samples: array with the sample positions
weights: function to compute the weight of each sample
"""
def Integrate(p, samples, weights):
	# ...
	pass



maximumSamples = 500
id = np.zeros(maximumSamples)
norm = np.zeros(maximumSamples)
poly = np.zeros(maximumSamples)

for i in range(maximumSamples):
	# id[i] =
	# norm[i] =
	# poly[i] =
	if i%10 == 0: # print progress
		print(i)

#plot results
plt.figure("Convergence")
# ...
