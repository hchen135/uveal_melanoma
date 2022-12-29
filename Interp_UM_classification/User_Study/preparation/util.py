from skimage.draw import circle_perimeter, line
from skimage.transform import resize
from skimage import img_as_ubyte
import numpy as np
from scipy.ndimage import binary_dilation
from copy import deepcopy

def convert_embedding_loc(embedding,image_shape):
	if embedding[0] < -1 or embedding[0] > 1 or embedding[1] < -1 or embedding[1] > 1:
		return None
	else:
		x = int(image_shape[0] - (embedding[1] + 1)/2*image_shape[1])
		y = int((embedding[0] + 1)/2*image_shape[0])
		return (x,y)

def embedding_counting_step(embedding,embedding_count):
	x = embedding[0]
	y = embedding[1]

	if x**2 + y**2 < 0.5**2:
		if x >= 0 and y >= 0:
			embedding_count[2] += 1
			return
		elif x >= 0 and y < 0:
			embedding_count[1] += 1
			return
		elif x < 0 and y >= 0:
			embedding_count[3] += 1
			return
		elif x < 0 and y < 0:
			embedding_count[0] += 1
			return
	else:
		if x >= 0 and y >= 0 and np.abs(x) >= np.abs(y):
			embedding_count[8] += 1
			return
		elif x >= 0 and y >= 0 and np.abs(x) < np.abs(y):
			embedding_count[9] += 1
			return
		elif x >= 0 and y < 0 and np.abs(x) >= np.abs(y):
			embedding_count[7] += 1
			return
		elif x >= 0 and y < 0 and np.abs(x) < np.abs(y):
			embedding_count[6] += 1
			return
		elif x < 0 and y >= 0 and np.abs(x) >= np.abs(y):
			embedding_count[11] += 1
			return
		elif x < 0 and y >= 0 and np.abs(x) < np.abs(y):
			embedding_count[10] += 1
			return
		elif x < 0 and y < 0 and np.abs(x) >= np.abs(y):
			embedding_count[4] += 1
			return
		elif x < 0 and y < 0 and np.abs(x) < np.abs(y):
			embedding_count[5] += 1
			return	

def embedding_add_partition(image):
	assert image.shape[0] == image.shape[1] 
	mask = np.zeros((image.shape[0],image.shape[1]))
	w = image.shape[1]
	h = image.shape[0]

	#draw circles
	large_rr, large_cc = circle_perimeter(w//2, h//2, w//2,shape=image.shape)
	large_rr_s, large_cc_s = circle_perimeter(w//2, h//2, w//2-1,shape=image.shape)

	small_rr, small_cc = circle_perimeter(w//2, h//2, w//4,shape=image.shape)
	small_rr_s, small_cc_s = circle_perimeter(w//2, h//2, w//4-1,shape=image.shape)

	mask[large_rr,large_cc] = 1
	mask[small_rr,small_cc] = 1
	mask[large_rr_s,large_cc_s] = 1
	mask[small_rr_s,small_cc_s] = 1

	#draw lines
	rr,cc = line(0,h//2,w-1,h//2)
	mask[rr,cc] = 1

	rr,cc = line(w//2,0,w//2,h-1)
	mask[rr,cc] = 1

	# bottom left line
	rr,cc = line(int(h/2+h/4/np.sqrt(2)),int(w/2 - w/4/np.sqrt(2)),int(h/2+h/2/np.sqrt(2)),int(w/2 - w/2/np.sqrt(2)))
	mask[rr,cc] = 1
	# bottom right line
	rr,cc = line(int(h/2+h/4/np.sqrt(2)),int(w/2 + w/4/np.sqrt(2)),int(h/2+h/2/np.sqrt(2)),int(w/2 + w/2/np.sqrt(2)))
	mask[rr,cc] = 1
	# top left line
	rr,cc = line(int(h/2-h/4/np.sqrt(2)),int(w/2 - w/4/np.sqrt(2)),int(h/2-h/2/np.sqrt(2)),int(w/2 - w/2/np.sqrt(2)))
	mask[rr,cc] = 1
	# top right line
	rr,cc = line(int(h/2-h/4/np.sqrt(2)),int(w/2 + w/4/np.sqrt(2)),int(h/2-h/2/np.sqrt(2)),int(w/2 + w/2/np.sqrt(2)))
	mask[rr,cc] = 1

	mask = binary_dilation(mask,iterations=1)

	image[mask>0] = [0,0,0]
	image[0,:] = [0,0,0]
	image[:,0] = [0,0,0]
	image[-1,:] = [0,0,0]
	image[:,-1] = [0,0,0]

	return image


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
