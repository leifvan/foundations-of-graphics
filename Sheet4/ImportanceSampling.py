import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# f: target function, which we want to integrate
def f(x):
    return x ** 0.9 * np.exp(-x ** 2 / 2)


# p_n: normal distribution for samples
sigma2 = 0.3
mu = 1.2


def p_n(x):
    return 1 / (np.sqrt(2 * np.pi * sigma2)) * np.exp(-(mu - x) ** 2 / (2 * sigma2))


# p_p: polynomial fit of the target function
fit_x = np.arange(0, 5)
fit_y = f(fit_x)
p_p = np.polynomial.Polynomial.fit(fit_x, fit_y, deg=4, domain=[0, 4])

# We expect p_u to perform worst as it's independent from the target function and doesnt
# take advantage of the particular distribution of f.
# p_n should perform significantly better as the gaussian is fitted to the function.
# Lastly p_p should perform best as it can fit multiple extrema unlike the gaussian.

# plot the function graphs:
plt.figure("Functions")
plot_x = np.linspace(0, 4, 80)
plt.plot(plot_x, f(plot_x), label="$f$")
plt.plot(plot_x, p_n(plot_x), label="$p_n$")
plt.plot(plot_x, p_p(plot_x), label="$p_p$")
plt.legend()
plt.show()


"""
Uses rejection sampling to generate samples
n: amount of samples to generate
d: distribution to draw samples from
max: maximum y value that is used to generate the samples
returns: x and y values of the samples in the shape (n, 2)
"""


def genSamples(n, d, max):
    s = np.zeros((n, 2))
    num_accepted = 0
    total = 0
    while num_accepted < n:
        points = np.random.random(size=(n - num_accepted, 2))
        total += len(points)
        points[:, 0] *= 4
        points[:, 1] *= max
        dist_vals = d(points[:, 0])
        points = points[points[:, 1] <= dist_vals]
        s[num_accepted:num_accepted + len(points)] = points
        num_accepted += len(points)
    return s, total


# Plot results of GenSamples()
# Hint: 0.8 is a reasonable value for the max parameter of GenSamples
plt.figure("Normal Distribution")
s, _ = genSamples(200, p_n, 0.8)
plt.plot(plot_x, p_n(plot_x), c="C1")
plt.scatter(s[:, 0], s[:, 1], s=3, c="C1")
plt.show()

plt.figure("Polynomial")
s, _ = genSamples(200, p_p, 0.8)
plt.plot(plot_x, p_p(plot_x), c="C2")
plt.scatter(s[:, 0], s[:, 1], s=3, c="C2")
plt.show()

"""
p: the function to integrate
samples: array with the sample positions
weights: function to compute the weight of each sample
"""
def integrate(p, samples, weights):
    return np.mean(p(samples) / weights(samples))

maximumSamples = 1500
id = np.zeros(maximumSamples)
norm = np.zeros(maximumSamples)
poly = np.zeros(maximumSamples)

# try to numerically integrate via trapezoidal method
trap_x = np.linspace(0, 4, 10000)
integral_pp = np.trapz(p_p(trap_x), trap_x)
integral_pn = np.trapz(p_n(trap_x), trap_x)

for i in range(1, maximumSamples):
    in_samples, total_samples = genSamples(i, lambda x: 1, 0.8)
    id[i] = integrate(f, in_samples[:, 0], lambda x: 1 / 4)
    in_samples, total_samples = genSamples(i, p_n, 0.8)
    norm[i] = integrate(f, in_samples[:, 0], lambda x: p_n(x) / integral_pn)
    in_samples, total_samples = genSamples(i, p_p, 0.8)
    poly[i] = integrate(f, in_samples[:, 0], lambda x: p_p(x) / integral_pp)
    if i % 10 == 0:  # print progress
        print(i)

# try to numerically integrate via trapezoidal method
trap_x = np.linspace(0, 4, 10*maximumSamples)
real_val = np.trapz(f(trap_x), trap_x)


# plot results
plt.figure("Convergence")
plt.plot([0,maximumSamples],[real_val, real_val], label="correct", c="black")
# smooth output with Savitzky-Golay filter
plt.plot(savgol_filter(id,11,3), label="uniform", alpha=0.8, lw=1)
plt.plot(savgol_filter(norm,11,3), label="normal", alpha=0.8, lw=1)
plt.plot(savgol_filter(poly,11,3), label="polynomial", alpha=0.8, lw=1)
plt.ylim([real_val-0.2, real_val+0.2])
plt.legend()
plt.show()
# c
# Everything worked exactly as we thought it would. )))

# d
# All kernels approach the same, correct, solution. The polynomial converges
# the quickest.
# In order for the integration to converge to the correct solution it is important that
# the sampled PDF integrates to 1 over the specified region, in this case [0, 4].
# In order to archive this we divide through the integral.



