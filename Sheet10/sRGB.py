import numpy as np
import matplotlib.pyplot as plt


XYZ_RGB_mat = np.array([[3.2406, -1.5372, -0.4986],
						[-0.9689, 1.8758, 0.0415],
						[0.0557, -0.204, 1.057]])
RGB_XYZ_mat = np.array([[0.4124, 0.3576, 0.1805],
						[0.2126, 0.7152, 0.0722],
						[0.0193, 0.1192, 0.9505]])

def XYZ_to_sRGB(c_xyz):
	# convert a single color with values in [0..1]
	# e.g. XYZ_to_sRGB([0.2, 0.1, 0.9]) yields [0.237, 0.194, 0.974]
	c_rgb = XYZ_RGB_mat @ c_xyz
	mask = c_rgb <= 0.0031308
	c_rgb[mask] *= 323/25
	c_rgb[~mask] = (211*c_rgb[~mask]**(5/12)-11)/200
	return np.clip(c_rgb, 0., 1.)


def sRGB_to_XYZ(c_rgb):
	# convert a single color with values in [0..1]
	# e.g. sRGB_to_XYZ([1, 0, 1]) yields [0.5932, 0.2848, 0.9788]
	c_xyz = np.copy(c_rgb)
	mask = c_xyz <= 0.04045
	c_xyz[mask] *= 25/323
	c_xyz[~mask] = ((200*c_xyz[~mask]+11)/211)**(12/5)
	c_xyz = RGB_XYZ_mat @ c_xyz
	return c_xyz




# sRGB interpolation:
gradient_sRGB = np.zeros((320, 3))
for i, a in enumerate(np.linspace(0, 1, gradient_sRGB.shape[0])):
	gradient_sRGB[i,:] = [a, 1-a, 0]


# linear interpolation:
gradient_lin = np.zeros((320, 3))
for i, a in enumerate(np.linspace(0, 1, gradient_lin.shape[0])):
	green_xyz = sRGB_to_XYZ([0.,1.,0.])
	red_xyz = sRGB_to_XYZ([1.,0.,0.])
	gradient_lin[i,:] = XYZ_to_sRGB(a*red_xyz+(1-a)*green_xyz)
	
plt.figure("interpolation comparisson")
plt.subplot(211)
plt.gca().set_title("sRGB")
plt.imshow(np.tile(gradient_sRGB, (100, 1, 1)))
plt.subplot(212)
plt.gca().set_title("linear")
plt.imshow(np.tile(gradient_lin, (100, 1, 1)))
plt.tight_layout()
plt.show()