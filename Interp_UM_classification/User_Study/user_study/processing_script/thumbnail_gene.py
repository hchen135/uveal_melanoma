from openslide import OpenSlide
from PIL import Image
import os

ORI_IMAGE_DIR = "/Users/hc/Documents/JHU/PJ/Mathias/microscopy/simple_guide"
slide_ind = 36

SlideRescaleDefault=512

SlideImg = OpenSlide(os.path.join(ORI_IMAGE_DIR,'Slide '+str(slide_ind)+'.svs'))

original_dimensions = SlideImg.dimensions
resize_width = original_dimensions[0]//SlideRescaleDefault
resize_height = original_dimensions[1]//SlideRescaleDefault

resized_SlideImg = SlideImg.get_thumbnail((resize_width,resize_height))
_width,_height = resized_SlideImg.size
resized_SlideImg = resized_SlideImg.resize((_width*3,_height*3),Image.NEAREST)
resized_SlideImg.save("../static/thumbnail/Slide "+str(slide_ind)+".jpg")

