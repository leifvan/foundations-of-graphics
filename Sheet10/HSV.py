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
	
	# IMPORTANT: Do not forget, to answer the questions from the assignment!
	# (You can answer them as code comments right here)
	# - Justify why you selected this color space
	# - Do you need all color channels for this?
	
	return clusters


# TODO: use either rgb or hsv color space:
	
# img_segmented = segment_by_threshold(img_rgb)
# img_segmented = segment_by_threshold(img_hsv)

plt.figure()
# we multiply the cluster colors with the original image to see, if the clustering is meaningful
plt.imshow(img_segmented * img_rgb)
plt.title('Segmented Image')
plt.show()
