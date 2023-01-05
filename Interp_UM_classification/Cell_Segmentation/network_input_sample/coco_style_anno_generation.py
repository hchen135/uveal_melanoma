import os
import json
import numpy as np
from glob import glob

def instance_segmentation_single_generation(good_anno,SLIC_map):
	list_axis_0 = []
	list_axis_1 = []
	for SLIC_ind in good_anno:
		x_ind,y_ind = np.where(SLIC_map == SLIC_ind)
		list_axis_0 = list_axis_0 + x_ind.tolist()
		list_axis_1 = list_axis_1 + y_ind.tolist()
	#bbox	
	min_axis_0 = min(list_axis_1)
	min_axis_1 = min(list_axis_0)
	max_axis_0 = max(list_axis_1)
	max_axis_1 = max(list_axis_0)
	return [list_axis_0, list_axis_1], [min_axis_0, min_axis_1, max_axis_0-min_axis_0, max_axis_1-min_axis_1], (max_axis_0-min_axis_0)*(max_axis_1-min_axis_1)

def SLIC_ind_flatten(SLIC_list):
	out = []
	for i in SLIC_list:
		for SLIC_ind in i:
			if SLIC_ind not in out:
				out.append(SLIC_ind)
	return out

def ind_to_loc(ind_list, SLIC_map):
	out = [[],[]]
	for ind in ind_list:
		x_ind,y_ind = np.where(SLIC_map == ind)
		x_ind = x_ind.tolist()
		y_ind = y_ind.tolist()		
		out[0] += x_ind
		out[1] += y_ind
	return out

def sementic_segmentation_extraction(SLIC_anno,SLIC_map):
	good_SLIC_ind = SLIC_ind_flatten(SLIC_anno['good'])
	bad_SLIC_ind = SLIC_ind_flatten(SLIC_anno['bad'])

	good_ind = ind_to_loc(good_SLIC_ind, SLIC_map)
	bad_ind = ind_to_loc(bad_SLIC_ind, SLIC_map)
	
	return {'good':good_ind,'bad':bad_ind}

def phase_recognition(img_path, train_list, val_list, test_list):
	if img_path[:-4] in train_list:
		return 'train'
	elif img_path[:-4] in val_list:
		return 'val'
	elif img_path[:-4] in test_list:
		return 'test'
	else:
		assert 0


img_list = glob('Slide */*.png')

images_train = []
annotations_train = []
images_val = []
annotations_val = []
images_test = []
annotations_test = []


with open('train.txt') as a:
	train_list = a.read().split('\n')
with open('val.txt') as a:
	val_list = a.read().split('\n')
with open('test.txt') as a:
	test_list = a.read().split('\n')

categories = [{'id':1,'name':'cell'}]

img_count = 1
segmentation_count = 1
for img_path in img_list:
	_phase = phase_recognition(img_path, train_list, val_list, test_list)
	print(img_path)
	images_single = {}
	images_single['file_name'] = img_path.split('/')[-2]+'/'+img_path.split('/')[-1]
	images_single['height'] = 256
	images_single['width'] = 256
	images_single['id'] = img_count
	if _phase == 'train':
		images_train.append(images_single)
	elif _phase == 'val':
		images_val.append(images_single)
	elif _phase == 'test':
		images_test.append(images_single)
	
	SLIC_map_path = img_path[:-4]+'.npy'
	SLIC_anno_path = img_path[:-4]+'.anno'
	
	SLIC_map = np.load(SLIC_map_path)
	with open(SLIC_anno_path) as _a:
		SLIC_anno = json.load(_a)
	
	images_single['semantic_seg_anno'] = sementic_segmentation_extraction(SLIC_anno,SLIC_map)
	num_good_instances_annotated = len(SLIC_anno['good'])
	if num_good_instances_annotated == 0 and _phase == 'train':
		print('*'*20+img_path.split('/')[-2]+'/'+img_path.split('/')[-1],img_count)
	for good_ind in range(num_good_instances_annotated):
		good_anno = SLIC_anno['good'][good_ind]
		if len(good_anno) > 0:
			annotations_single = {}
			annotations_single['iscrowd'] = 0
			annotations_single['image_id'] = img_count
			annotations_single['category_id'] = 1
			annotations_single['id'] = segmentation_count
			
			segmentation_single, bbox_single, area_single = instance_segmentation_single_generation(good_anno,SLIC_map)
			
			annotations_single['segmentation'] = segmentation_single
			annotations_single['bbox'] = bbox_single
			annotations_single['area'] = area_single
			if _phase == 'train':
				annotations_train.append(annotations_single)	
			if _phase == 'val':
				annotations_val.append(annotations_single)	
			if _phase == 'test':
				annotations_test.append(annotations_single)	
	
			segmentation_count += 1

	img_count += 1

ANNO_train = {}
ANNO_train['annotations'] = annotations_train
ANNO_train['images'] = images_train
ANNO_train['categories'] = categories

ANNO_val = {}
ANNO_val['annotations'] = annotations_val
ANNO_val['images'] = images_val
ANNO_val['categories'] = categories

ANNO_test = {}
ANNO_test['annotations'] = annotations_test
ANNO_test['images'] = images_test
ANNO_test['categories'] = categories
'''
with open('coco_style_anno_train.json','w') as a:
	json.dump(ANNO_train,a,indent=4)
with open('coco_style_anno_val.json','w') as a:
	json.dump(ANNO_val,a,indent=4)
with open('coco_style_anno_test.json','w') as a:
	json.dump(ANNO_test,a,indent=4)
'''
