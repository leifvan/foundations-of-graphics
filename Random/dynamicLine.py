import matplotlib.pyplot as plt
import time
import random
import numpy as np
from scipy import signal
fig = plt.figure()
position = np.array([
    np.arange(0, 100), np.zeros(100)])
velocity=np.zeros(98)
force = np.zeros((98))

xdata = []
ydata = []

plt.show()
axes = plt.gca()
axes.set_xlim(0, 100)
axes.set_ylim(-5, +5)
line, = axes.plot(xdata, ydata, 'r-')

# print(np.convolve(np.array([1,2,3,-2]),np.array([0.5,0,0.5]),"same"))
""

i=0
while(1):
    i+=1
    line.set_xdata(position[0, :])
    line.set_ydata(position[1, :])
    position[1,0]=np.sin(i/15)
    # position[1,2]=-2
    force=-1*(position[1,1:-1]-(0.5*(position[1,:-2]+position[1,2:])))
    # force=- (position[1,1:-1]-(np.convolve(position[1,:],np.array([1,0,1]),"same")[1:-1]*0.5))

    velocity+=force
    position[1, 1:-1]+=velocity
    fig.canvas.flush_events()
    plt.draw()
    # plt.pause(1e-17)
    time.sleep(0.0006)
#
