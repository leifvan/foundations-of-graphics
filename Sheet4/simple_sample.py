import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

start = 0
end = 4

# f: target function, which we want to integrate
def f(x):
    return x ** 0.9 * np.exp(-x ** 2 / 2)


# p_n: normal distribution for samples
sigma2 = 0.3
mu = 1.2

def p_n(x):
    return 1 / (np.sqrt(2 * np.pi * sigma2)) * np.exp(-(mu - x) ** 2 / (2 * sigma2))

fit_x = np.arange(0, 5)
fit_y = f(fit_x)
p_p = np.polynomial.Polynomial.fit(fit_x, fit_y, deg=4, domain=[0, 4])


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


def get_samples(shape):
    return np.random.random(shape)*4

def integrate(samples, weights):
    sum = 0
    for i in range(samples.shape[0]):
        sum += f(samples[i]) / weights(samples[i])
    return sum / samples.shape[0]

num = 5000

print(integrate(get_samples(num), lambda x: 1 / (end-start)))
samples_pn, _ = genSamples(num, p_n, 0.8)
print(integrate(samples_pn[:, 0], p_n))
samples_pp, _ = genSamples(num, p_p, 0.8)
trap_x = np.linspace(0, 4, num)
real_val = np.trapz(p_p(trap_x), trap_x)
print(real_val)
print(integrate(samples_pp[:, 0], lambda x: p_p(x) / real_val))
