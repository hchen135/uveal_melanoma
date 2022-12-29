import json
import os
from shutil import copyfile

slide_ind = 65

UMAP_EMBEDDING_INFO_DIR = "/Users/hc/Documents/JHU/PJ/Mathias/microscopy/cell_segmentation/result/11242020_all_image/umap_3000_circle"
TILE_IMG_DIR = "/Users/hc/Documents/Uveal Melanoma/Uveal Melanoma ROI extracted"

SAVE_DIR = "../user_study/static/cell_tiles"


if not os.path.exists(os.path.join(SAVE_DIR,"Slide "+str(slide_ind))):
    os.mkdir(os.path.join(SAVE_DIR,"Slide "+str(slide_ind)))
    os.mkdir(os.path.join(SAVE_DIR,"Slide "+str(slide_ind),"raw"))
    os.mkdir(os.path.join(SAVE_DIR,"Slide "+str(slide_ind),"origin"))
    os.mkdir(os.path.join(SAVE_DIR,"Slide "+str(slide_ind),"zoomed"))

with open(os.path.join(UMAP_EMBEDDING_INFO_DIR,'umap_proj_'+str(slide_ind)+'_info.json')) as a:
    content = json.load(a)

for ind in range(len(content['file_name'])):
	file_name = content['file_name'][ind]
	img_ori_path = os.path.join(TILE_IMG_DIR,file_name)

	dest_path = os.path.join(SAVE_DIR,"Slide "+str(slide_ind),"raw",file_name.split('/')[-1])
	if not os.path.exists(dest_path):
		copyfile(img_ori_path,dest_path)