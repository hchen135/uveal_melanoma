from openslide import OpenSlide
from PIL import Image
import numpy as np
from skimage.io import imsave
import os

slide_ind = 65
h_min = 250
h_max = 550
w_min = 400
w_max = 900

ORI_IMAGE_DIR = "../user_study/static/slides"

SlideRescaleDefault=512

SlideImg = OpenSlide(os.path.join(ORI_IMAGE_DIR,'Slide '+str(slide_ind)+'.svs'))

original_dimensions = SlideImg.dimensions
resize_width = original_dimensions[0]//SlideRescaleDefault
resize_height = original_dimensions[1]//SlideRescaleDefault

resized_SlideImg = SlideImg.get_thumbnail((resize_width,resize_height))
_width,_height = resized_SlideImg.size
resized_SlideImg = resized_SlideImg.resize((_width*3,_height*3),Image.NEAREST)
resized_SlideImg = np.array(resized_SlideImg)
resized_SlideImg = resized_SlideImg[h_min:h_max,w_min:w_max]

imsave("../user_study/static/thumbnail/Slide "+str(slide_ind)+"_task3.jpg",resized_SlideImg)

#############
# 13

# Task 1: 
# h_min = 0
# h_max = 400
# w_min = 0
# w_max = 600

# Task 2: 
# h_min = 50
# h_max = 450
# w_min = 280
# w_max = 880

# Task 3:
# h_min = 50
# h_max = 450
# w_min = 100
# w_max = 700

#############
# 24

# Task 1: 
# h_min = 0
# h_max = 450
# w_min = 0
# w_max = 700

# Task 2: 
# h_min = 0
# h_max = 450
# w_min = 300
# w_max = 1000

# Task 3: 
# h_min = 150
# h_max = 500
# w_min = 500
# w_max = 1000

#############
# 29

# Task 1: 
# h_min = 0
# h_max = 450
# w_min = 0
# w_max = 600

# Task 2: 
# h_min = 100
# h_max = 550
# w_min = 270
# w_max = 870

# Task 3:
# h_min = 100
# h_max = 450
# w_min = 480
# w_max = 980

#############
# 51

# Task 1: 
# h_min = 0
# h_max = 300
# w_min = 0
# w_max = 500

# Task 2: 
# h_min = 0
# h_max = 500
# w_min = 400
# w_max = 700

# Task 3:
# h_min = 100
# h_max = 550
# w_min = 600
# w_max = 1080

#############
# 59

# Task 1: 
# h_min = 0
# h_max = 370
# w_min = 0
# w_max = 600

# Task 2: 
# h_min = 0
# h_max = 370
# w_min = 200
# w_max = 800

# Task 3:
# h_min = 300
# h_max = 550
# w_min = 250
# w_max = 700

#############
# 65

# Task 1: 
# h_min = 0
# h_max = 250
# w_min = 450
# w_max = 900

# Task 2: 
# h_min = 100
# h_max = 450
# w_min = 250
# w_max = 700

# Task 3:
# h_min = 250
# h_max = 550
# w_min = 400
# w_max = 900



