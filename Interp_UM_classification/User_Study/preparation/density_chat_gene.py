import matplotlib.pyplot as plt 
import numpy as np 
import json
from skimage.io import imsave
import os

from util import convert_embedding_loc,embedding_add_partition,embedding_counting_step
slide_ind = 65

UMAP_EMBEDDING_INFO_DIR = "/Users/hc/Documents/JHU/PJ/Mathias/microscopy/cell_segmentation/result/11242020_all_image/umap_3000_circle"
EMBEDDING_SHAPE = (512,512,3)
COLOR = [255,0,0]

with open(os.path.join(UMAP_EMBEDDING_INFO_DIR,'umap_proj_'+str(slide_ind)+'_info.json')) as a:
	content = json.load(a)

embedding_loc = content['embedding_loc']

image = np.ones(EMBEDDING_SHAPE)*255


embedding_counting = np.zeros(12)
for embedding in embedding_loc:
	converted_embedding = convert_embedding_loc(embedding,image.shape)
	if converted_embedding is not None:
		image[converted_embedding[0],converted_embedding[1]] = COLOR
	embedding_counting_step(embedding,embedding_counting)
image = embedding_add_partition(image)
image = image.astype(np.uint8)
for i in range(12):
	print(i+1,np.round(embedding_counting[i]/np.sum(embedding_counting),3))
imsave('../user_study/static/pie_chart/Slide '+str(slide_ind)+'.jpg',image)
