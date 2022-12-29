import sys
import numpy as np
from skimage.io import imread,imsave
import os
from glob import glob
from copy import deepcopy
from skimage.transform import resize
from skimage import img_as_ubyte

def zoom_fullres_name_locate(request_json,slide_img,status=""):
	clientY = request_json.get("clientY")
	clientX = request_json.get("clientX")
	pointX = request_json.get("pointX")
	pointY = request_json.get("pointY")
	scale = request_json.get("scale")
	window_width = request_json.get("window_width")
	window_height = request_json.get("window_height")
	oriY = request_json.get("oriY")
	oriX = request_json.get("oriX")
	oriScale = request_json.get("oriScale")  
	slide_ind = request_json.get("slide_ind")
	_thumbnail_width = request_json.get("_thumbnail_width")
	_thumbnail_height = request_json.get("_thumbnail_height")
	_thumbnail_ori_width = request_json.get("_thumbnail_ori_width")
	_thumbnail_ori_height = request_json.get("_thumbnail_ori_height")
	globalX = request_json.get("globalX")
	globalY = request_json.get("globalY")
	# left_margin_value = globalX*scale + pointX
	# top_margin_value = globalY*scale + pointY
	left_margin_value = globalX
	top_margin_value = globalY

	local_axis_x = clientX - left_margin_value
	local_axis_y = clientY - top_margin_value
	print (local_axis_x < oriX*scale,local_axis_y < oriY*scale,local_axis_x > (window_width - oriX)*scale,local_axis_y > (window_height - oriY)*scale)
	print(globalX,globalY,local_axis_x,oriX*scale,(window_width - oriX)*scale,local_axis_y,oriY*scale,(window_height - oriY)*scale)
	if local_axis_x < oriX*scale or local_axis_y < oriY*scale or local_axis_x > (window_width - oriX)*scale or local_axis_y > (window_height - oriY)*scale:
		# TODO output the blank image
		# return "/static/blank_image.png"
		return ""
	else:
		true_x = int((local_axis_x - oriX*scale)/scale/oriScale)
		true_y = int((local_axis_y - oriY*scale)/scale/oriScale)
		if status == "rotate90":
			tmp = true_y
			true_y = _thumbnail_width - true_x
			true_x = tmp
			
		true_x = true_x + _thumbnail_ori_width
		true_y = true_y + _thumbnail_ori_height
		tile_x = true_x // 3
		tile_y = true_y // 3
		patch_x = int(true_x - tile_x * 3)
		patch_y = int(true_y - tile_y * 3)
		print("tile_x:",tile_x,", tile_y:",tile_y,", patch_x:",patch_x,", patch_y:",patch_y,status,file=sys.stdout)


		candidate_name = "TileLoc_"+str(tile_y)+"_"+str(tile_x)+"_"+str(patch_x)+"_"+str(patch_y)+".png"

		# generate the image
		# first delete previous generated image.
		OUTPUT_DIR = "./user_study/static/UM_tiles"		
		if not os.path.exists(os.path.join(OUTPUT_DIR,"Slide "+str(slide_ind))):
			os.mkdir(os.path.join(OUTPUT_DIR,"Slide "+str(slide_ind)))

		previous_generated_img_list = glob(os.path.join(OUTPUT_DIR,"Slide "+str(slide_ind),"*"))
		for _previous_img in previous_generated_img_list:
			os.remove(_previous_img)

		# generate the image
		tile = slide_img.read_region((int(tile_x*512 + patch_x*128),int(tile_y*512 + patch_y*128)),0,(256,256))
		tile.save(os.path.join(OUTPUT_DIR,"Slide "+str(slide_ind),candidate_name))

		return "/static/UM_tiles/Slide "+str(slide_ind)+"/"+candidate_name
		#if candidate_name in tile_list:
		#	return "/static/UM_tiles/Slide "+str(slide_ind)+"/"+candidate_name
		#else:
		#	# return "/static/blank_image.png"
		#	return ""
	# print("clientX:",inputs["clientX"] - 50,", pointX:",inputs["pointX"],", calculated:",(50)*inputs["scale"] + inputs["pointX"],file=sys.stdout)

def dist(coord1,coord2):
	return (coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2

def find_first_k_smallest(coord,pie_chart_dict,k=3):
	dist_all = [dist(coord,coord2) for coord2 in pie_chart_dict['embedding_loc']]
	print(len(pie_chart_dict['embedding_loc']))

	index = np.argpartition(dist_all,range(k))[:k]

	# image_names = [pie_chart_dict['file_name'][i] for i in index]
	# embedding_locs = [pie_chart_dict['embedding_loc'][i] for i in index]

	# for i in embedding_locs:
	# 	print(i)

	# TODO
	# some points may be too far away.
	return index


def tile_img_bbox_gene(img,bbox,dilation_iter=3,color=[0,255,0]):
    img_tmp = deepcopy(img)
    mask = np.zeros((img_tmp.shape[0],img_tmp.shape[1]))
    y1,x1,y2,x2 = bbox
    x1 = int(np.clip(x1,0,255))
    x1_enlarge = int(np.clip(x1-dilation_iter,0,255))
    x2 = int(np.clip(x2,0,255))
    x2_enlarge = int(np.clip(x2+dilation_iter,0,255))
    y1 = int(np.clip(y1,0,255))
    y1_enlarge = int(np.clip(y1-dilation_iter,0,255))
    y2 = int(np.clip(y2,0,255))
    y2_enlarge = int(np.clip(y2+dilation_iter,0,255))
    mask[x1_enlarge:x2_enlarge,y1_enlarge:int(y1+1)] = 1
    mask[x1_enlarge:x2_enlarge,y2:int(y2_enlarge+1)] = 1
    mask[x1_enlarge:int(x1+1),y1_enlarge:y2_enlarge] = 1
    mask[x2:int(x2_enlarge+1),y1_enlarge:y2_enlarge] = 1

    img_tmp[mask>0] = color

    return img_tmp

def cell_img_bbox_gene(img,bbox,dilation_iter=3,color=[0,255,0]):
    # first determine the area to crop
    y1,x1,y2,x2 = bbox
    h = x2 - x1
    w = y2 - y1
    if h*3/2 > img.shape[0] or w*3/2 > img.shape[1]:
        return tile_img_bbox_gene(img,bbox,dilation_iter,color)
    s = max(h,w)
    crop_area_h = int(max(s*3/2,img.shape[0]//3))
    crop_area_w = int(max(s*3/2,img.shape[1]//3))

    crop_area_x1 = int((x1+x2)/2 - crop_area_h/2)
    crop_area_x2 = int(crop_area_x1 + crop_area_h)
    crop_area_y1 = int((y1+y2)/2 - crop_area_w/2)
    crop_area_y2 = int(crop_area_y1 + crop_area_w)
    # print('1: ',crop_area_y1,crop_area_x1,crop_area_y2,crop_area_x2)
    if crop_area_x1 < 0:
        crop_area_x1 = 0
        crop_area_x2 = int(crop_area_h)
    if crop_area_x2 >= img.shape[0]:
        crop_area_x2 = int(img.shape[0]-1)
        crop_area_x1 = int(img.shape[0]-1 - crop_area_h)
    if crop_area_y1 < 0:
        crop_area_y1 = 0
        crop_area_y2 = int(crop_area_w)
    if crop_area_y2 >= img.shape[1]:
        crop_area_y2 = int(img.shape[1]-1)
        crop_area_y1 = int(img.shape[1]-1 - crop_area_w) 
    # print('2: ',crop_area_y1,crop_area_x1,crop_area_y2,crop_area_x2)                                                   
    # Then determine the cropped image and new bbox
    img_cropped = img[crop_area_x1:crop_area_x2,crop_area_y1:crop_area_y2]
    # print(img_cropped.shape)
    assert img_cropped.shape[0] == img_cropped.shape[1]
    img_cropped = resize(img_cropped,img.shape)
    img_cropped = img_as_ubyte(img_cropped)
    ratio = img.shape[0]/(crop_area_h)
    x1_new = (x1-crop_area_x1)*ratio
    x2_new = (x2-crop_area_x1)*ratio
    y1_new = (y1-crop_area_y1)*ratio
    y2_new = (y2-crop_area_y1)*ratio
    # print(y1_new,x1_new,y2_new,x2_new)
    # print(crop_area_y1,crop_area_x1,crop_area_y2,crop_area_x2)
    # print(y1,x1,y2,x2)
    bbox_new = [y1_new,x1_new,y2_new,x2_new]
    # Finally draw
    return tile_img_bbox_gene(deepcopy(img_cropped).astype(np.uint8),bbox_new,dilation_iter,color)






def pie_chart_display_prep(request_json,pie_chart_dict,status="description"):
	# first convert the coordinates to local coordinates
	slide_ind = request_json.get("slide_ind")
	clientY = request_json.get("clientY")
	clientX = request_json.get("clientX")
	window_width = request_json.get("window_width")
	window_height = request_json.get("window_height")
	globalX = request_json.get("globalX")
	globalY = request_json.get("globalY")

	if window_width < window_height:
		localY = (window_height - window_width)/2
		localX = 0
		size = window_width
	else:
		localY = 0
		localX = (window_width - window_height)/2
		size = window_height

	if status == "description":
		# print('window',clientX,clientY,window_width,window_height,localX,localY,size)
		if clientY - globalY > localY and clientY - globalY < localY + size and clientX - globalX > localX and clientX - globalX < localX + size:
			relateX = clientX - globalX - localX
			relateY = clientY - globalY - localY

			coordX = relateX
			coordY = size - relateY

			coordX = coordX/size*2 - 1
			coordY = coordY/size*2 - 1

			print(clientX,clientY,relateX,relateY,coordX,coordY)
			index = find_first_k_smallest([coordX,coordY],pie_chart_dict,3)
			img_names = []

			# generate the image
			# first delete previous generated image.
			OUTPUT_DIR = "./user_study/static/cell_tiles"		

			previous_generated_img_list = glob(os.path.join(OUTPUT_DIR,"Slide "+str(slide_ind),"origin","*")) + glob(os.path.join(OUTPUT_DIR,"Slide "+str(slide_ind),"zoomed","*"))
			for _previous_img in previous_generated_img_list:
				os.remove(_previous_img)


			for i in range(len(index)):
				img_names.append("/static/cell_tiles/Slide "+str(slide_ind)+"/origin/"+str(index[i])+".jpg")
				img_names.append("/static/cell_tiles/Slide "+str(slide_ind)+"/zoomed/"+str(index[i])+".jpg")

				img = imread(os.path.join(OUTPUT_DIR,"Slide "+str(slide_ind),"raw",pie_chart_dict['file_name'][index[i]].split('/')[-1]))
				tile_img_bbox = tile_img_bbox_gene(img,pie_chart_dict['boxes'][index[i]])
				cell_img_bbox = cell_img_bbox_gene(img,pie_chart_dict['boxes'][index[i]])
				imsave(os.path.join(OUTPUT_DIR,"Slide "+str(slide_ind),"origin",str(index[i])+".jpg"),tile_img_bbox)
				imsave(os.path.join(OUTPUT_DIR,"Slide "+str(slide_ind),"zoomed",str(index[i])+".jpg"),cell_img_bbox)


			while len(img_names) < 6:
				img_names.append("/static/blank_image.png")
			print(img_names)
			return img_names
		else:
			return ""











