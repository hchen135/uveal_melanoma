from openslide import OpenSlide
from PIL import Image
import os
import numpy as np 
# from skimage.io import imsave
# from skimage.transform import resize

slide_ind = 59

INPUT_DIR = "../user_study/static/slides"
OUTPUT_DIR = "../user_study/static/UM_tiles"

SlideRescaleDefault=512

if not os.path.exists(os.path.join(OUTPUT_DIR,"Slide "+str(slide_ind))):
	os.mkdir(os.path.join(OUTPUT_DIR,"Slide "+str(slide_ind)))



SlideImg = OpenSlide(os.path.join(INPUT_DIR,'Slide '+str(slide_ind)+'.svs'))

original_dimensions = SlideImg.dimensions
num_width = original_dimensions[0]//SlideRescaleDefault
num_height = original_dimensions[1]//SlideRescaleDefault

print(num_width,num_height)
for w in range(num_width):
	for h in range(num_height):
		for patch_h in range(3):
			for patch_w in range(3):
				tile = SlideImg.read_region((int(w*512 + patch_w*128),int(h*512 + patch_h*128)),0,(256,256))
				tile = tile.resize((128,128))
				tile.save(os.path.join(OUTPUT_DIR,"Slide "+str(slide_ind),"_".join(["TileLoc",str(h),str(w),str(patch_h),str(patch_w)])+".png"))
				
					



