import numpy as np
import matplotlib.pyplot as plt


def XYZ_to_sRGB(c_xyz):
	# convert a single color with values in [0..1]
	# e.g. XYZ_to_sRGB([0.2, 0.1, 0.9]) yields [0.237, 0.194, 0.974]
	# TODO: implement...
	
	return c_rgb


def sRGB_to_XYZ(c_rgb):
	# convert a single color with values in [0..1]
	# e.g. sRGB_to_XYZ([1, 0, 1]) yields [0.5932, 0.2848, 0.9788]
	# TODO: implement...
	
	return c_xyz




# sRGB interpolation:
gradient_sRGB = np.zeros((320, 3))
for i, a in enumerate(np.linspace(0, 1, gradient_sRGB.shape[0])):
	gradient_sRGB[i,:] = [a, 1-a, 0]


# linear interpolation:
gradient_lin = np.zeros((320, 3))
for i, a in enumerate(np.linspace(0, 1, gradient_lin.shape[0])):
	# TODO: adapt this line to interpolate colors in XYZ space
	gradient_lin[i,:] = [a, 1-a, 0]
	
plt.figure("interpolation comparisson")
plt.subplot(211)
plt.gca().set_title("sRGB")
plt.imshow(np.tile(gradient_sRGB, (100, 1, 1)))
plt.subplot(212)
plt.gca().set_title("linear")
plt.imshow(np.tile(gradient_lin, (100, 1, 1)))
plt.tight_layout()

