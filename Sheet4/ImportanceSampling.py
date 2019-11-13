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


def GenSamples(n, d, max):
    s = np.zeros((n, 2))
    num_accepted = 0
    while num_accepted < n:
        points = np.random.random(size=(n - num_accepted, 2))
        points[:, 0] *= 4
        points[:, 1] *= max
        dist_vals = d(points[:, 0])
        points = points[points[:, 1] <= dist_vals]
        s[num_accepted:num_accepted + len(points)] = points
        num_accepted += len(points)
    return s


# Plot results of GenSamples()
# Hint: 0.8 is a reasonable value for the max parameter of GenSamples
plt.figure("Normal Distribution")
s = GenSamples(200, p_n, 0.8)
plt.plot(plot_x, p_n(plot_x), c="C1")
plt.scatter(s[:, 0], s[:, 1], s=3, c="C1")
plt.show()

plt.figure("Polynomial")
s = GenSamples(200, p_p, 0.8)
plt.plot(plot_x, p_p(plot_x), c="C2")
plt.scatter(s[:, 0], s[:, 1], s=3, c="C2")
plt.show()

"""
p: the function to integrate
samples: array with the sample positions
weights: function to compute the weight of each sample
"""


def Integrate(p, samples, weights):
    weighted_samples = p(samples) / weights(samples)
    return weighted_samples.mean()


maximumSamples = 500
id = np.zeros(maximumSamples)
norm = np.zeros(maximumSamples)
poly = np.zeros(maximumSamples)

for i in range(1, maximumSamples):
    id[i] = Integrate(f, GenSamples(i, lambda x: 1, 1)[:, 0], lambda x: 1/4)
    norm[i] = Integrate(f, GenSamples(i, p_n, 0.8)[:, 0], p_n)
    poly[i] = Integrate(f, GenSamples(i, p_p, 0.8)[:, 0], p_p)
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
