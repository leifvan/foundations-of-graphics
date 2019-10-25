import numpy as np
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
from scipy import signal
from time import time

figure=plt.figure()
mask=plt.imread("./mask.png").sum(axis=2)
# mask= np.zeros((7,7))
position= np.zeros(mask.shape)
velocity= np.zeros(mask.shape)

graph=plt.imshow(position,vmin=-2,vmax=2)
plt.colorbar()
radius=1
i = 0
timeCalculation=0
timeDrawing=0

while (1):
    t=time()
    i += 1
    position[mask.shape[0]//2-radius:mask.shape[0]//2+radius,mask.shape[1]//2-radius:mask.shape[1]//2+radius] = -np.sin(i / 15)*8
    force=-(position-signal.convolve2d(position,[[0,1,0],[1,0,1],[0,1,0]],mode="same")/4)
    velocity+=force
    timeCalculation+=time()-t
    t=time()
    position+=velocity
    position[mask < 1] = 0
    graph.set_data(position)
    figure.canvas.draw()
    figure.canvas.flush_events()
    # plt.draw()
    timeDrawing+=time()-t
    if i%100==0:
        print("Average calculation time %f"%(timeCalculation/100))
        print("Average drawing time %f"%(timeDrawing/100))