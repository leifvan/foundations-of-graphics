import matplotlib.pylab as plt
from skimage.color import rgb2hsv #from the scikit-image package
import numpy as np


# load 'peppers.png'
# the values are given in uint8, i.e. [255,0,0] represents 'red'
img_rgb = plt.imread('peppers.png')

# the HSV image can be calculated using for example skimage 'rgb2hsv':
img_hsv = rgb2hsv(img_rgb);

# display the images:
plt.figure()
plt.imshow(img_rgb)
plt.title('1. Original image in RGB space')
plt.figure()
plt.imshow(img_hsv)
plt.title('2. Converted to HSV space')

# PART A
# convert the HSV image back to the RGB space
def convert_hsv2rgb(img_hsv):
	img_rgb = None # return value

	# your code goes here...
	original_shape = img_hsv.shape
	img_hsv = np.reshape(img_hsv, newshape=(-1,3))
	h,s,v = np.rollaxis(img_hsv, axis=-1)
	chroma = v * s
	h_prime = h * 6
	x = chroma * (1-np.abs(h_prime % 2 - 1))
	img_rgb = np.zeros_like(img_hsv)

	zero = np.zeros_like(h_prime)
	perms = [[chroma, x, zero], [x, chroma, zero], [zero, chroma, x],
			 [zero, x, chroma], [x, zero, chroma], [chroma, zero, x]]
	for lb, perm in enumerate(perms):
		mask = (lb <= h_prime) & (h_prime < (lb+1))
		img_rgb[mask] = np.stack([perm[0][mask], perm[1][mask], perm[2][mask]], axis=1)
	img_rgb += (v - chroma)[:,None]
	img_rgb.shape = original_shape
	return img_rgb


img_RGB_estimated = convert_hsv2rgb(img_hsv)

# display the result:
plt.figure()
plt.imshow(img_RGB_estimated)
plt.title('3. re-conversion of HSV image to RGB space')





# PART B
# segment the image based on thresholding of the color information:
def segment_by_threshold(img):
	# this function returns another image, with the same size as the original
	# but with only two different colors, representing the clusters
	clusters = np.zeros(img.shape)
	
	# your code goes here...
	img = np.copy(img)
	h, s, v = np.rollaxis(img, axis=2)

	hue_mask = ~((h > 0.5) & (h < 0.93))
	value_mask = v > 0.9

	clusters[hue_mask | value_mask] = 1

	# IMPORTANT: Do not forget, to answer the questions from the assignment!
	# (You can answer them as code comments right here)
	# - Justify why you selected this color space
	# we chose hsv, because the background has a specific hue that is not present
	# in the peppers. This means, in hsv color space we only need to threshold the
	# hue to separate the background.

	# - Do you need all color channels for this?
	# No, we did not use the saturation channel from hsv. We did use the value channel
	# to recover highlights in the peppers, but for a basic thresholding the hue
	# channel would have been enough.

	return clusters



# img_segmented = segment_by_threshold(img_rgb)
img_segmented = segment_by_threshold(img_hsv)

plt.figure()
# we multiply the cluster colors with the original image to see, if the clustering is meaningful
plt.imshow(img_segmented * img_rgb)
plt.title('Segmented Image')
plt.show()
