import numpy as np 
import os
import torch
import json
from shutil import copyfile
from skimage.io import imsave
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.morphology import binary_dilation
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
import random
from copy import deepcopy
from scipy.stats import mode
import time
import multiprocessing
import cv2

from config import config 
from config_cervical import config as config_cervical

def config_selection(dataset):
        if dataset == "UM":
                return config
        elif dataset == "Cervical":
                return config_cervical

def store_config(config):
	if not os.path.exists(os.path.join(config['out_dir'],config['date'])):
		os.mkdir(os.path.join(config['out_dir'],config['date']))
	copyfile('config.py', os.path.join(config['out_dir'],config['date'],'config.py'))


def check_ckpt_dir():
	checkpoint_dir = os.path.join('..','checkpoint')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

def is_best_ckpt(epoch,loss_tr,loss_cv,config):
	check_ckpt_dir()
	best_json = os.path.join('..','checkpoint',config['date']+'.json')
	best_loss_cv = best_loss_tr = float("Inf")

	if os.path.exists(best_json):
		with open(best_json) as infile:
			data = json.load(infile)
			best_loss_cv = data['loss_cv']
			best_loss_tr = data['loss_tr']

	if loss_cv < best_loss_cv:
		with open(best_json,'w') as outfile:
			json.dump({
				'epoch': epoch,
				'loss_tr': loss_tr,
				'loss_cv': loss_cv,
				}, outfile)
		return True
	return False

def save_ckpt(model, optimizer, epoch, loss_tr, loss_cv, config):
	def do_save(filepath):
		torch.save({
			'epoch': epoch,
			'name': config['model_name'],
			'date': config['date'],
			'learning_rate': config['learning_rate'],
			'model': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			}, filepath)
	# check if best_checkpoint
	if is_best_ckpt(epoch,loss_tr,loss_cv,config):
		filepath=os.path.join('..','checkpoint',config['date']+'.pkl')
		do_save(filepath)


def _extract_state_from_dataparallel(checkpoint_dict):
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in checkpoint_dict.items():
		name = k
		new_state_dict[name] = v
	return new_state_dict

def load_ckpt(model=None,filepath=None):
	print(filepath)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint = torch.load(filepath,map_location=device)
	epoch = checkpoint['epoch']
	learning_rate = checkpoint['learning_rate']
	optimizer = checkpoint['optimizer']
	
	if model:
		
		model_dict = model.state_dict()
		pretrain_dict = checkpoint['model']
		detect_dict = {k: v for k,v in pretrain_dict.items() if k in model_dict}
		full_dict = {}
		#print(list(model_dict.items())[0][0],list(pretrain_dict.items())[0][0])
		
		for k, v in model_dict.items():
			for k_, v_ in pretrain_dict.items():
				if k == k_ or k == k_.split('module.')[-1] or k_ == k.split('module.')[-1]:
					full_dict[k] = v_
					print(k,'FOUND')
			if k not in full_dict:
				full_dict[k] = v
				print(k,'NOT FOUND')
		
		model.load_state_dict(_extract_state_from_dataparallel(full_dict))
		full_dict = None
		checkpoint = None
	return epoch, learning_rate, optimizer

def record_epoch(config,epoch):
	val_log = open("../log/val_"+config['date']+".txt","a")
	val_log.write('epoch '+str(epoch)+'\n')
	val_log.close()

	
def surrounding_loc_generation(loc,shape, _iter=3):
	h,w = shape
	output = []
	for l in loc:
		for _h in range(max(0,l[0]-_iter),min(h,l[0]+_iter)):
			for _w in range(max(0,l[1]-_iter),min(w,l[1]+_iter)):
				if [_h,_w] not in loc and [_h,_w] not in output:
					output.append([_h,_w])
	return output

def SLIC_single(batch_ind,SLIC_label,ind_mask,config,SLIC_cluster_dict,SLIC_area_dict):
	SLIC_cluster = np.zeros_like(ind_mask)
	rps = regionprops(SLIC_label+1)
	SLIC_area_list = {}
	_count = 0
	for rp in rps:
		loc = rp.coords
		values = ind_mask[loc[:,0],loc[:,1]]
		mode_value = mode(values)[0][0]
		SLIC_cluster[loc[:,0],loc[:,1]] = mode_value
			
		area = np.zeros_like(ind_mask)
		area[loc[:,0],loc[:,1]] = 1
		h_min = np.min(loc[:,0])
		h_max = np.max(loc[:,0])
		w_min = np.min(loc[:,1])
		w_max = np.max(loc[:,1])
		area_dilated = np.zeros((h_max-h_min+6,w_max-w_min+6))
		area_dilated[loc[:,0]-h_min+3,loc[:,1]-w_min+3] = 1
		area_dilated = binary_dilation(binary_dilation(binary_dilation(area_dilated)))-area_dilated
		rp_surrounding = regionprops(area_dilated.astype(np.uint8))
		assert len(rp_surrounding) == 1
		loc_surrounding = rp_surrounding[0].coords
		loc_surrounding[:,0] = loc_surrounding[:,0]+h_min - 3
		loc_surrounding[:,1] = loc_surrounding[:,1]+w_min - 3
		shape_0 = loc_surrounding.shape[0]
		loc_surrounding = loc_surrounding[loc_surrounding[:,0] >=0,:]	
		shape_1 = loc_surrounding.shape[0]
		loc_surrounding = loc_surrounding[loc_surrounding[:,1] >=0,:]	
		shape_2 = loc_surrounding.shape[0]
		loc_surrounding = loc_surrounding[loc_surrounding[:,0] <ind_mask.shape[0],:]	
		shape_3 = loc_surrounding.shape[0]
		loc_surrounding = loc_surrounding[loc_surrounding[:,1] <ind_mask.shape[1],:]	
		shape_4 = loc_surrounding.shape[0]
		loc = loc.transpose().tolist()
		loc_surrounding = loc_surrounding.transpose().tolist()
		SLIC_area_list[_count] = [loc,loc_surrounding]
		_count+=1
		
	SLIC_cluster_dict[batch_ind] = SLIC_cluster
	SLIC_area_dict[batch_ind] = SLIC_area_list

def SLIC_batch(SLIC_label,ind_mask,config):
	print(ind_mask.shape)
	ind_mask = ind_mask.detach().cpu().numpy()
	result_cluster = np.zeros_like(ind_mask)
	
	manager = multiprocessing.Manager()
	SLIC_cluster_dict = manager.dict()
	SLIC_area_dict = manager.dict()
	pool = multiprocessing.Pool(config['batch_size'])
	for i in range(ind_mask.shape[0]):
		pool.apply_async(SLIC_single, args=(i,SLIC_label[i],ind_mask[i],config,SLIC_cluster_dict,SLIC_area_dict))
	pool.close()
	pool.join()
	for i in range(ind_mask.shape[0]):
		result_cluster[i] = SLIC_cluster_dict[i]
	return result_cluster, SLIC_area_dict

'''
def SLIC_batch(SLIC_label,ind_mask,config,device):
	for batch_ind in range(ind_mask.shape[0]):
		label_img = SLIC_label[batch_ind]
		time1 = time.time()
		ind_mask_batch = ind_mask[batch_ind]
		for i in torch.unique(label_img):
			mask = label_img == i
			#mode_value = torch.mode(ind_mask_batch[loc[:,0],loc[:,1]])[0]
			ind_mask_batch[mask] = torch.mode(ind_mask_batch[mask])[0]
		print('Single Assignment time: ',time.time()-time1)
	return ind_mask
'''


def switch_learning_rate(learning_rate,loss_stored,loss_val,switch_learning_rate_interval = 5):
	if len(loss_stored) == 0:
		loss_stored = [loss_val]
		return learning_rate,loss_stored
	if loss_val < min(loss_stored):
		loss_stored = []
	if len(loss_stored) < switch_learning_rate_interval:
		loss_stored.append(loss_val)
	else:
		loss_stored = []
		loss_stored.append(loss_val)
		learning_rate = learning_rate*0.1

	return learning_rate, loss_stored

def test_save_img(epoch,assignment, output,config, phase):
	out_dir = config['out_dir']
	epoch = str(epoch)
	name = config['date']
	new_assignment = {}
	new_output = {}
	_sample = torch.tensor([0])
	if not os.path.exists(os.path.join(out_dir,name)):
		os.mkdir(os.path.join(out_dir,name))
	if not os.path.exists(os.path.join(out_dir,name,epoch)):
		os.mkdir(os.path.join(out_dir,name,epoch))
	for i in assignment:
		slide_num = i.split('/')[-2]
		tile_num = i.split('/')[-1]
		target = assignment[i]
		img_output = output[i]
		if 'bagnet' in phase:
			target = target.tolist()
		else:
			target = float(target)
		new_assignment[slide_num + '/'+tile_num] = target
		new_output[slide_num + '/'+tile_num] = img_output.tolist()
	with open(os.path.join(out_dir,name,epoch,'pred.txt'),'w') as outfile:
		json.dump(new_assignment,outfile,indent=4)
	with open(os.path.join(out_dir,name,epoch,'pred_vector.txt'),'w') as outfile:
		json.dump(new_output,outfile,indent=4)


def test_save_cell_segmentation(epoch,semantic_segmentation_output,instance_segmentation_output,config):
	out_dir = config['out_dir']
	epoch = str(epoch)
	name = config['date']
	if not os.path.exists(os.path.join(out_dir,name)):
		os.mkdir(os.path.join(out_dir,name))
	if not os.path.exists(os.path.join(out_dir,name,epoch)):
		os.mkdir(os.path.join(out_dir,name,epoch))
	if not os.path.exists(os.path.join(out_dir,name,epoch,'semantic')):
		os.mkdir(os.path.join(out_dir,name,epoch,'semantic'))
	if not os.path.exists(os.path.join(out_dir,name,epoch,'instance')):
		os.mkdir(os.path.join(out_dir,name,epoch,'instance'))
	
	for i in semantic_segmentation_output:
		slide_name, img_name = i.split('/')[-2:]
		if not os.path.exists(os.path.join(out_dir,name,epoch,'semantic',slide_name)):
			os.mkdir(os.path.join(out_dir,name,epoch,'semantic',slide_name))
		if not os.path.exists(os.path.join(out_dir,name,epoch,'instance',slide_name)):
			os.mkdir(os.path.join(out_dir,name,epoch,'instance',slide_name))
		np.save(os.path.join(out_dir,name,epoch,'semantic',slide_name,img_name),np.array(semantic_segmentation_output[i]).astype(np.uint8))
		np.save(os.path.join(out_dir,name,epoch,'instance',slide_name,img_name),np.array(instance_segmentation_output[i]).astype(np.int8))
def label_select(center_1_ind,center_2_ind,label_loc,center_dict):
	center_1 = center_dict[center_1_ind]
	center_2 = center_dict[center_2_ind]
	dist_1 = (center_1[0]-label_loc[0])**2+(center_1[1]-label_loc[1])**2
	dist_2 = (center_2[0]-label_loc[0])**2+(center_2[1]-label_loc[1])**2
	if dist_1 < dist_2:
		return center_1_ind
	else:
		return center_2_ind

def l2_dist(sample,criterion):
	output = []
	for i in range(len(sample)):
		output.append(np.sum((sample[i]-criterion)**2))
	return output

def otsu(values):#n
	ordered_values = np.sort(values)
	statistics = []#n-1, represent how many to remain -1
	for i in range(1,len(ordered_values)-1):
		W1 = i
		W2 = len(ordered_values)-i
		mu1 = np.average(ordered_values[:i])
		mu2 = np.average(ordered_values[i:])
		stat = W1*W2*(mu1-mu2)**2
		statistics.append(stat)
	return ordered_values[np.argmax(statistics)]

def instance_prediction_single(feature_sample,labels_area_sample,batch_ind,output_dict):
	center_dict = {}
	output_sample = (np.ones((feature_sample.shape[-2],feature_sample.shape[-1]))*(-1))
	for supx_ind in range(len(labels_area_sample)):
		# extract features needed for k-means
		loc = labels_area_sample[supx_ind][0]
		loc_surrounding = labels_area_sample[supx_ind][1]
		loc_calc = [loc[0]+loc_surrounding[0],loc[1]+loc_surrounding[1]]
		features_consider = feature_sample[:,loc_calc[0],loc_calc[1]].transpose()
		# k-means
		mean_feature = np.average(features_consider,axis=0)
		dist = l2_dist(features_consider,mean_feature)
		thresh = otsu(dist)
		preserve_bool = dist <= thresh
		#kmeans = KMeans(n_clusters=2, random_state=0,max_iter=10).fit(features_consider)
		#labels = kmeans.labels_
		#mode_label = mode(labels[:len(loc[0])])[0][0]
		#preserve_bool = labels == mode_label
		# center information
		loc_preserved = np.array(loc_calc)[:,preserve_bool]
		center_h = np.average(loc_preserved[0])
		center_w = np.average(loc_preserved[1])
		center_dict[supx_ind] = [center_h,center_w]
		# insert labels to output and check if it is obtained.
		#print(supx_ind)
		output_sample[loc_preserved[:,0],loc_preserved[:,1]] = supx_ind
		for pixel_ind in range(loc_preserved.shape[1]):
			target_h = loc_preserved[0,pixel_ind]
			target_w = loc_preserved[1,pixel_ind]
			if output_sample[target_h,target_w] == -1:
				output_sample[target_h,target_w] = supx_ind
			else:
				previous_label = output_sample[target_h,target_w]	
				#print(previous_label)
				label_selected = label_select(previous_label,supx_ind,[target_h,target_w],center_dict)
				output_sample[target_h,target_w] = label_selected
		
	output_dict[batch_ind] = output_sample

def instance_prediction(feature,labels_area,config):
	output = (np.ones((feature.shape[0],feature.shape[-2],feature.shape[-1]))*(-1))

	manager = multiprocessing.Manager()
	output_dict = manager.dict()
	pool = multiprocessing.Pool(config['batch_size'])
	time1 = time.time()
	for i in range(feature.shape[0]):
		pool.apply_async(instance_prediction_single, args=(feature[i],labels_area[i],i,output_dict))
	pool.close()
	pool.join()

	for i in range(feature.shape[0]):
		output[i] = output_dict[i]
	print(time.time()-time1)
	return output



def instance_prediction_one_time(feature,labels_area,config):
	output = (np.ones((feature.shape[0],feature.shape[-2],feature.shape[-1]))*(-1))
	for batch_ind in range(feature.shape[0]):
		feature_sample = feature[batch_ind]
		labels_area_sample = labels_area[batch_ind]
		center_dict = {}
		for supx_ind in range(len(labels_area_sample)):
			# extract features needed for k-means
			loc = labels_area_sample[supx_ind][0]
			loc_surrounding = labels_area_sample[supx_ind][1]
			#print(loc)
			#print(loc_surrounding)
			loc_calc = [loc[0]+loc_surrounding[0],loc[1]+loc_surrounding[1]]
			features_consider = feature_sample[:,loc_calc[0],loc_calc[1]].transpose()
			# k-means
			kmeans = KMeans(n_clusters=2, random_state=0).fit(features_consider)
			labels = kmeans.labels_
			# preserve center labels
			mode_label = mode(labels[:len(loc[0])])[0][0]
			preserve_bool = labels == mode_label
			# center information
			loc_preserved = np.array(loc_calc)[:,preserve_bool]
			center_h = np.average(loc_preserved[0])
			center_w = np.average(loc_preserved[1])
			center_dict[supx_ind] = [center_h,center_w]
			# insert labels to output and check if it is obtained.
			print(supx_ind)
			output[batch_ind,loc_preserved[:,0],loc_preserved[:,1]] = supx_ind
			'''
			for pixel_ind in range(loc_preserved.shape[1]):
				target_h = loc_preserved[0,pixel_ind]
				target_w = loc_preserved[1,pixel_ind]
				
				if output[batch_ind,target_h,target_w] == -1:
					output[batch_ind,target_h,target_w] = supx_ind
				else:
					previous_label = output[batch_ind,target_h,target_w]
					#print(previous_label)
					label_selected = label_select(previous_label,supx_ind,[target_h,target_w],center_dict)
					output[batch_ind,target_h,target_w] = label_selected
			'''
	return output		

def naive_slide_aggregation_auc(pred_all,slide_ind_all,class_dict,config):
	if config['n_class'] == 3:
		class_convertion_dict = {0:0,1:1,2:2}
	elif config['n_class'] == 2:
		class_convertion_dict = {0:0,1:0,2:1}
	else:
		class_convertion_dict={}
		for i in range(config['n_class']):
			class_convertion_dict[int(i)] = int(i)	

	slide_pred_dict = {}
	slide_label_dict = {}
	slide_available = np.unique(slide_ind_all)
	labels_all = []
	slide_pred_all = []
	for slide_ind in slide_available:
		pred_used = np.array(pred_all)[np.array(slide_ind_all) == slide_ind,:]				
		slide_pred_dict[slide_ind] = np.average(pred_used,axis=0)
		slide_label_dict[slide_ind] = class_convertion_dict[class_dict[str(slide_ind)]['class_ind']]
		labels_all.append(class_convertion_dict[class_dict[str(slide_ind)]['class_ind']])
		slide_pred_all.append(np.average(pred_used,axis=0))
	
	naive_aggregated_auc = []
	for i in range(config['n_class']):
		naive_aggregated_auc.append(roc_auc_score((np.array(labels_all) == i).astype(np.uint8),np.array(slide_pred_all)[:,i].astype(np.float32)))
	
	pred_corr_label = [slide_pred_all[i][labels_all[i]] for i in range(len(labels_all))]
	acc = np.average(np.array(pred_corr_label) > 0.5)
	return naive_aggregated_auc, acc, slide_label_dict, slide_pred_dict

def slide_level_feature_aggregation(w_feature_all,pred_all,slide_num_all,pred_softmax_all, threshold=100, fake_num_threshold = 1000000, phase='train'):
	slide_level_feature = {}
	slide_level_pred_naiive = {}
	slide_ind = np.unique(slide_num_all)
	for ind in range(len(slide_ind)):
		bool_vector = torch.tensor(slide_num_all == slide_ind[ind]).float().unsqueeze(1).unsqueeze(2)	
		print('Slide '+str(slide_ind[ind])+' has '+str(torch.sum(bool_vector).item())+' samples')
		
		if phase == 'train' and int(torch.sum(bool_vector).item()) > threshold:
			for fake_ind in range(min(fake_num_threshold,int(int(torch.sum(bool_vector).item())//threshold))):
				#rand_criterion = torch.max(torch.tensor([0.1,threshold/torch.sum(bool_vector).item()]))
				rand_criterion = torch.tensor(threshold/torch.sum(bool_vector).item())
				rand_prob = torch.rand(1)*(1-rand_criterion)+rand_criterion
				rand_tensor = (torch.rand_like(bool_vector.squeeze(1).squeeze(1))<rand_prob).float()
				used_tensor = bool_vector*rand_tensor.unsqueeze(1).unsqueeze(1)
				print('used_tensor.shape',used_tensor.shape)
				print('bool_vector.shape',bool_vector.shape)
				print('pred_softmax_all.shape',pred_softmax_all.shape)
	
				w_feature_sum = torch.sum(w_feature_all*used_tensor,dim=0)
				pred_sum = torch.sum(pred_all*used_tensor.squeeze(1),dim=0).unsqueeze(1)
				averaged_pred = torch.mean(pred_softmax_all[used_tensor[:,0,0].to(torch.bool),:],dim=0)
				averaged_feature  = w_feature_sum/pred_sum
				#normalization
				#normalized_feature = (averaged_feature-torch.mean(averaged_feature,1,True))/torch.std(averaged_feature,1).unsqueeze(1)
				#assert len(normalized_feature.shape) == 2
				slide_level_feature[str(slide_ind[ind])+'_'+str(fake_ind)] = averaged_feature#[n_class,2048]
				slide_level_pred_naiive[str(slide_ind[ind])+'_'+str(fake_ind)] = averaged_pred.cpu().numpy().astype(np.float32).tolist()
		if 1:
			w_feature_sum = torch.sum(w_feature_all*bool_vector,dim=0)
			pred_sum = torch.sum(pred_all*bool_vector.squeeze(1),dim=0).unsqueeze(1)
			averaged_pred = torch.mean(pred_softmax_all[bool_vector[:,0,0].to(torch.bool),:],dim=0)
			averaged_feature  = w_feature_sum/pred_sum
			slide_level_feature[str(slide_ind[ind])] = averaged_feature#[n_class,2048]
			slide_level_pred_naiive[str(slide_ind[ind])] = averaged_pred.cpu().numpy().astype(np.float32).tolist()

	return slide_level_feature, slide_level_pred_naiive

def batch_generation(slide_level_feature_dict,dataset,class_dict,class_representation_dict,config,shuffle=False):
	feature_out = []
	label_out = []
	slide_id_out = []
	available_slide_ind = list(slide_level_feature_dict.keys())
	if shuffle:
		random_ind = np.random.choice(available_slide_ind,len(available_slide_ind),replace=False).tolist()
	else:
		random_ind = available_slide_ind

	batch_size = config['batch_size_second_stage']
	for i in range(int(len(random_ind)//batch_size)):
		feature = []
		label = []
		slide_id = []
		for j in range(batch_size):
			slide_id_single = random_ind[i*batch_size+j]
			if dataset == 'UM':
				slide_id.append(slide_id_single)
			elif dataset == 'Cervical':
				slide_id.append(slide_id_single)
			feature.append(slide_level_feature_dict[slide_id_single])
			if dataset == 'UM':
				label.append(class_representation_dict[int(class_dict[str(slide_id_single.split('_')[0])]['class_ind'])])
			elif dataset == 'Cervical':
				label.append(class_representation_dict[int(class_dict[slide_id_single]['class_ind'])])
		feature = torch.stack(feature,dim=0)
		feature_out.append(feature)
		label_out.append(label)
		slide_id_out.append(slide_id)
	if len(random_ind) % batch_size >1:
		feature = []
		label = []
		slide_id = []
		for j in range(int(len(random_ind)//batch_size*batch_size),len(random_ind)):
			slide_id_single = random_ind[j]
			if dataset == 'UM':
				slide_id.append(slide_id_single)
			elif dataset == 'Cervical':
				slide_id.append(slide_id_single)
			if dataset == 'UM':
				label.append(class_representation_dict[int(class_dict[str(slide_id_single.split('_')[0])]['class_ind'])])	
			elif dataset == 'Cervical':
				label.append(class_representation_dict[int(class_dict[slide_id_single]['class_ind'])])	
			feature.append(slide_level_feature_dict[slide_id_single])
		feature = torch.stack(feature,dim=0)
		feature_out.append(feature)
		label_out.append(label)
		slide_id_out.append(slide_id)
	return feature_out, label_out, slide_id_out
def heatmap_generation(heatmap,img_name,config):
	slide_name = [i.split('/')[-2] for i in img_name]
	tile_name = [i.split('/')[-1] for i in img_name]
	inputs = [cv2.imread(i) for i in img_name]
	
	if config['if_train_first_stage']:
		heatmap_path = '../heatmap/'+config['date']
	else:
		heatmap_path = '../heatmap/'+config['first_stage_pretrain_model']
	if not os.path.exists(heatmap_path):
		os.mkdir(heatmap_path)
	
	for ind,heatmap_single in enumerate(heatmap):
		input_single = inputs[ind]
		if not os.path.exists(os.path.join(heatmap_path,slide_name[ind])):
			os.mkdir(os.path.join(heatmap_path,slide_name[ind]))
		if not os.path.exists(os.path.join(heatmap_path,slide_name[ind],tile_name[ind].split('.')[0]+'_'+str(int(0))+'.'+tile_name[ind].split('.')[1])):
			for i in range(config['n_class']):
				heatmap_choose = heatmap_single[i,:,:]
				np.save(os.path.join(heatmap_path,slide_name[ind],tile_name[ind].split('.')[0]+'_'+str(int(i))+'.npy'),heatmap_choose)
				heatmap_choose = resize(heatmap_choose,(config['img_size'],config['img_size']))
				heatmap_choose = cv2.cvtColor((heatmap_choose*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
				heatmap_choose = cv2.applyColorMap(heatmap_choose, cv2.COLORMAP_JET)
				output = cv2.addWeighted(heatmap_choose, 0.3, input_single, 0.7, 0)
				cv2.imwrite(os.path.join(heatmap_path,slide_name[ind],tile_name[ind].split('.')[0]+'_'+str(int(i))+'.'+tile_name[ind].split('.')[1]),output)
			copyfile(img_name[ind], os.path.join(heatmap_path,slide_name[ind],tile_name[ind]))

def feature_generation(feature_dict, heatmap, feature, label, threshold=0.6):
	def insert(feature_dict, feature, label, class_index):
		try:
			feature_dict[class_index] += feature.tolist()
			feature_dict['label_'+str(class_index)] += label.tolist()
			#feature_dict[class_index] = np.concatenate([feature_dict[class_index],feature],axis=0)
		except:
			feature_dict[class_index] = feature.tolist()
			feature_dict['label_'+str(class_index)] = label.tolist()

	response_diff = heatmap[:,0,:,:] - heatmap[:,1,:,:]
	# create pixel label
	pixel_label = np.repeat(label[:,np.newaxis],int(feature.shape[2]*feature.shape[3]),axis=1).reshape(-1)
	feature = feature.transpose(0,2,3,1)
	feature = feature.reshape(-1,feature.shape[-1])
	response_diff_1 = (response_diff > threshold).reshape(-1)
	if np.sum(response_diff_1) > 0:
		feature_1 = feature[response_diff_1]
		label_1 = pixel_label[response_diff_1]
		print('feature extracted 1: ', feature_1.shape)
		insert(feature_dict,feature_1,label_1,0)	

	response_diff_2 = (response_diff < -threshold).reshape(-1)
	if np.sum(response_diff_2) > 0:
		feature_2 = feature[response_diff_2]
		label_2 = pixel_label[response_diff_2]
		print('feature extracted 2: ', feature_2.shape)
		insert(feature_dict,feature_2,label_2,1)
	
	return feature_dict
	
def feature_store(feature_dict,config,phase):
	print("START STORING FEATURE")
	if config['if_train_first_stage']:
		heatmap_path = '../heatmap/'+config['date']
	else:
		heatmap_path = '../heatmap/'+config['first_stage_pretrain_model']
	print(heatmap_path)
	with open(heatmap_path+'/feature_dict_'+phase+'.json','w') as a:
		json.dump(feature_dict,a,indent=4)

def store_heatmap_info(general_heatmap_info, detail_heatmap_info, y_sigmoid, img_names):
	for sample_ind in range(len(img_names)):
		img_path = img_names[sample_ind]
		heatmap = y_sigmoid[sample_ind].astype(np.float32).tolist()
		
		slide_name = img_path.split('/')[-2]
		img_name = img_path.split('/')[-1]
		
		general_heatmap = np.average(y_sigmoid[sample_ind],axis=(1,2)).tolist()
		
		if slide_name not in general_heatmap_info:
			general_heatmap_info[slide_name] = {}
		if slide_name not in detail_heatmap_info:
			detail_heatmap_info[slide_name] = {}

		general_heatmap_info[slide_name][img_path] = general_heatmap
		detail_heatmap_info [slide_name][img_path] = heatmap
