from math import sin, cos, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt

s = 4
t = 0
h = pi / s
p = np.array([1,0], dtype=np.float)
v = np.zeros(2)

x = np.zeros((s+1,2))


def acc(t):
    return np.array([np.cos(t), np.sin(t)])


def xt(t):
    return np.array([-np.cos(t)+2, t - np.sin(t)])


for i in range(s+1):
    # print("t =", t)
    # print("p =", p)
    # print("v =", v)
    # print("a =", acc(t))
    # print("-" * 4)
    x[i] = p
    v += acc(t)*h
    p += v*h
    t += h

print("t =", t)
print("p =", p)
print("v =", v)
print("a =", acc(t))
print("-" * 4)

print(np.array([1,0]) + pi**2 / 16 * np.array([1,0]))
print(np.array([1,0]) + pi**2 / 16 * np.array([2+1/sqrt(2), 1/sqrt(2)]))
print(np.array([1,0]) + pi**2 / 16 * np.array([3+2/sqrt(2), 1+2/sqrt(2)]))
print(np.array([1,0]) + pi**2 / 16 * np.array([4+2/sqrt(2), 2+4/sqrt(2)]))

plt.plot([1, 1 + pi**2], [0, 0], '-x', label=r'Euler $h=\pi$')
plt.plot(x[:,0], x[:,1], '-x', label=r'Euler $h=\frac{\pi}{4}$')

a = xt(np.linspace(0,pi,50))
print(a.shape)
plt.plot(a[0], a[1], label="Analytical")
plt.plot(a[0,(0,49)], a[1,(0,49)], 'x', color='C2')

plt.legend()
plt.tight_layout()
plt.show()