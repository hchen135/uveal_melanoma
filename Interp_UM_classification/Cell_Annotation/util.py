import torch
import numpy as np 

lambda_1 = 0.1 # (L1-L2)/min(L1,L2) < lambda_1 is treated as the borderline sample

def good_bad_cluster_selection(selection_dict):
	good = []
	bad = []
	for i in selection_dict:
		if np.average(selection_dict[i]) < 0.3:
			bad.append(int(i))
		if np.average(selection_dict[i]) > 0.7:
			good.append(int(i))
	return good,bad

def check_good_bad_boundary(ind,good_cluster,bad_cluster): # sample by sample
	assert len(ind) == 2
	good = False
	bad = False
	for i in ind:
		if i in good_cluster:
			good = True
		elif i in bad_cluster:
			bad = True
	return (good and bad)
def check_distance_condition(dist,lambda_1): # array in total (3*3*batch_size,x)
	min_,_ = torch.min(dist,1)
	ratio = torch.abs(dist[:,0] - dist[:,1])/min_
	return ratio < lambda_1

def batch_borderline_sample_detection(M,name_use,output_dict,good_cluster,bad_cluster,lambda_1):
	M = torch.tensor(M).float()
	output_use = torch.tensor([output_dict[name] for name in name_use]).float()
	output_use = output_use.reshape(-1,output_use.shape[-1]) # (3*3*batch_size,x)
	print(torch.sum((torch.tensor(M).unsqueeze(0)-output_use.unsqueeze(2))**2,dim=1).shape)
	dist,ind = torch.sort(torch.sum((torch.tensor(M).unsqueeze(0)-output_use.unsqueeze(2))**2,dim=1),dim=1)
	dist = dist[:,:2]
	ind = ind[:,:2]
	_bool1 = torch.tensor([check_good_bad_boundary(i,good_cluster,bad_cluster) for i in ind])
	_bool2 = check_distance_condition(dist,lambda_1)
	_bool = _bool1*_bool2
	return _bool.numpy().tolist() #batch_size*3*3

def borderline_sample_detection(M,output_dict,selection_dict,lambda_1,batch_size = 10000):
	# M (x, 100) dim
	# output_dict: {name:[3,3,x],...}
	good_cluster,bad_cluster = good_bad_cluster_selection(selection_dict)
	image_names = [i for i in output_dict]
	_bool_all = []
	total_iter = len(output_dict)//batch_size
	for _iter in range(total_iter):
		name_use = image_names[_iter*batch_size:(_iter+1)*batch_size]
		_bool_all += batch_borderline_sample_detection(M,name_use,output_dict,good_cluster,bad_cluster,lambda_1)
	name_use = image_names[total_iter*batch_size:]
	_bool_all += batch_borderline_sample_detection(M,name_use,output_dict,good_cluster,bad_cluster,lambda_1)
	#print(_bool_all.shape)
	_bool_all = torch.tensor(_bool_all).reshape(-1,3,3)
	print(_bool_all.shape)
	_bool_all = _bool_all.numpy()
	name_ind,height_ind,width_ind = np.where(_bool_all)

	name_selected = [image_names[i] for i in name_ind]
	height_ind = height_ind.tolist()
	width_ind = width_ind.tolist()

	return list(zip(name_selected,height_ind,width_ind)),good_cluster,bad_cluster

def patch_image_extract(slide,height_num,width_num):
	tile_height_num = height_num//3
	tile_width_num = width_num//3

	patch_height_num = height_num - tile_height_num*3
	patch_width_num = width_num - tile_width_num*3

	return slide.read_region((int(tile_width_num*512+patch_width_num*128),int(tile_height_num*512+patch_height_num*128)),0,(256,256))






