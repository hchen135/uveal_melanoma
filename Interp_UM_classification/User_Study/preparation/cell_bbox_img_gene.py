import json
import os
from skimage.io import imread,imsave

from util import tile_img_bbox_gene,cell_img_bbox_gene

slide_ind = 65

UMAP_EMBEDDING_INFO_DIR = "/Users/hc/Documents/JHU/PJ/Mathias/microscopy/cell_segmentation/result/11242020_all_image/umap_3000_circle"
TILE_IMG_DIR = "/Users/hc/Documents/Uveal Melanoma/Uveal Melanoma ROI extracted"

SAVE_DIR = "../user_study/static/cell_tiles"

if not os.path.exists(os.path.join(SAVE_DIR,"Slide "+str(slide_ind))):
    os.mkdir(os.path.join(SAVE_DIR,"Slide "+str(slide_ind)))
    os.mkdir(os.path.join(SAVE_DIR,"Slide "+str(slide_ind),"origin"))
    os.mkdir(os.path.join(SAVE_DIR,"Slide "+str(slide_ind),"zoomed"))

with open(os.path.join(UMAP_EMBEDDING_INFO_DIR,'umap_proj_'+str(slide_ind)+'_info.json')) as a:
    content = json.load(a)

for ind in range(len(content['file_name'])):
    file_name = content['file_name'][ind]
    mask_size = content['mask_size'][ind]
    bbox = content['boxes'][ind]
    embedding_loc = content['embedding_loc'][ind]

    img = imread(os.path.join(TILE_IMG_DIR,file_name))

    tile_img_w_bbox = tile_img_bbox_gene(img,bbox,dilation_iter=3)
    zoom_img_w_bbox = cell_img_bbox_gene(img,bbox,dilation_iter=3)

    imsave(os.path.join(SAVE_DIR,"Slide "+str(slide_ind),"origin",str(ind)+'.jpg'),tile_img_w_bbox)
    imsave(os.path.join(SAVE_DIR,"Slide "+str(slide_ind),"zoomed",str(ind)+'.jpg'),zoom_img_w_bbox)


