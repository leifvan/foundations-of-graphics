import matplotlib.pyplot as plt
import matplotlib.image
import scipy.signal
import numpy as np

# this is a convenience function for nicer image plotting with adjusted colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plotImg(img, title=None, ax=None, **more):
    if ax is None:
        ax = plt.gca()
    i = ax.imshow(img, **more)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(i, cax=cax)
    return ax


# Part A
def MyConvolve2d(f, g):
    pad_0, pad_1 = g.shape[0], g.shape[1]
    pad_half_0, pad_half_1 = pad_0 // 2, pad_1 // 2
    c = np.zeros_like(f,shape=(f.shape[0]+pad_0,f.shape[1]+pad_1))
    f = np.pad(f, ((pad_0, pad_0), (pad_1, pad_1)), mode='wrap')
    g = np.flip(g, axis=0)

    for i in range(f.shape[0] - pad_0):
        for j in range(f.shape[1] - pad_1):
            f_patch = f[i:i + pad_0, j:j + pad_1]
            c[i ,j] = np.sum(np.multiply(f_patch, g))

    return c[1:,:-1]


# Part B
def MyFftConcolve2d(f, g):
    pad_0, pad_1 = g.shape[0], g.shape[1]
    f = np.pad(f, ((pad_0, pad_0), (pad_1, pad_1)), mode='wrap')
    f_freq, g_freq = np.fft.rfft2(f), np.fft.rfft2(g, s=f.shape)
    fg_freq = f_freq * g_freq
    c = np.fft.irfft2(fg_freq)
    return c[pad_0:-1,pad_1:-1]


def PartA():
    k = matplotlib.image.imread('kernel.png')[:, :, 0]
    i = matplotlib.image.imread('img.png')[:, :, 0]
    c = scipy.signal.convolve2d(i, k, boundary='wrap')
    m = MyConvolve2d(i, k)

    plt.figure("Convolution by Hand")

    plt.subplot(141)
    plotImg(i, "Original Image", interpolation="nearest", cmap=plt.cm.gray)

    plt.subplot(142)
    plotImg(c, "Python Convolution", interpolation="nearest", cmap=plt.cm.gray)

    plt.subplot(143)
    plotImg(m, "Convolution by Hand", interpolation="nearest", cmap=plt.cm.gray)

    if m.shape != c.shape:
        print("Image Dimensions do not match!")
    else:
        plt.subplot(144)
        plotImg(m - c, "Difference", interpolation="nearest", cmap=plt.cm.gray)

    plt.show()


def PartB():
    i = matplotlib.image.imread('img.png')[:, :, 0]
    k = matplotlib.image.imread('kernel.png')[:, :, 0]
    c = scipy.signal.convolve2d(i, k, boundary='wrap')
    m = MyFftConcolve2d(i, k)

    plt.figure("Convolution with FFT")

    plt.subplot(141)
    plotImg(i, "Original Image", interpolation="nearest", cmap=plt.cm.gray)

    plt.subplot(142)
    plotImg(c, "Python Convolution", interpolation="nearest", cmap=plt.cm.gray, vmax=32)

    plt.subplot(143)
    plotImg(m, "FFT Convolution", interpolation="nearest", cmap=plt.cm.gray, vmax=32)

    if m.shape != c.shape:
        print("Image Dimensions do not match!")
    else:
        plt.subplot(144)
        plotImg(m - c, "Difference", interpolation="nearest", cmap=plt.cm.gray)

    plt.show()


PartA()
#PartB()
