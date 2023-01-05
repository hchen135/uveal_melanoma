# We only needs 2 parts: class 1 umap gene slides, other test slides.
import json
from glob import glob
import numpy as np

# first choose 30 random class 1 clides as umap gene slides.
#umap_gene_slides = np.random.choice(range(1,51), 30, replace=False).astype(int).tolist()
umap_gene_slides = np.random.choice(range(51,101), 20, replace=False).astype(int).tolist()
print(umap_gene_slides)
#umap_gene_slides = [ 3,  8, 29, 11, 21, 16, 35, 28,  2, 18, 24, 34, 19,  9, 13, 46, 45, 12, 30, 33]
other_slides = [i for i in range(1,101) if i not in umap_gene_slides]

def img_list_gene(slides):
    img_list = []
    for i in slides:
        img_list += glob('../../Uveal Melanoma ROI extracted/Slide '+str(i)+'/*.png')
    print(len(img_list))
    return img_list

umap_gene_img_list = img_list_gene(umap_gene_slides)
other_img_list = img_list_gene(other_slides)

def coco_style_info_gene(img_list):
    img_count = 1
    imgs = []
    for img_path in img_list:
        img_single = {}
        img_single['file_name'] = img_path.split('/')[-2]+'/'+img_path.split('/')[-1]
        img_single['height'] = 256
        img_single['width'] = 256
        img_single['id'] = img_count
        imgs.append(img_single)
        img_count += 1
    return imgs

umap_gene_coco_style = coco_style_info_gene(umap_gene_img_list)
other_coco_style = coco_style_info_gene(other_img_list)

ANNO_umap_gene = {}
ANNO_umap_gene['images'] = umap_gene_coco_style

ANNO_other = {}
ANNO_other['images'] = other_coco_style

with open('class2_umap_gene_coco_style.json','w') as a:
    json.dump(ANNO_umap_gene,a,indent=4)
with open('class2_other_slides_coco_style.json','w') as a:
    json.dump(ANNO_other,a,indent=4)



