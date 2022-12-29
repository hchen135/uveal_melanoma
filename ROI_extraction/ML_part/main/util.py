import numpy as np 
import os
import torch
import json
from shutil import copyfile
from skimage.io import imsave
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.cluster import KMeans
import random

from config import config 
from config_ctf import config as config_ctf
from config_bagnet import config as config_bagnet
from config_bagnet_ctf import config as config_bagnet_ctf
from config_test import config as config_test

def config_selection(phase):
	if phase == "normal":
		return config
	elif phase == "coarse_to_fine":
		return config_ctf
	elif phase == "bagnet":
		return config_bagnet
	elif not phase is None and 'coarse_to_fine' in phase and 'bagnet' in phase:
		return config_bagnet_ctf
	else:
		return config_test
def store_config(config,phase):
	if not os.path.exists(os.path.join(config['out_dir'],config['date'])):
		os.mkdir(os.path.join(config['out_dir'],config['date']))
	if phase == "normal":	
		copyfile('config.py', os.path.join(config['out_dir'],config['date'],'config.py'))
	elif phase == "coarse_to_fine":
		copyfile('config_ctf.py', os.path.join(config['out_dir'],config['date'],'config_ctf.py'))
	elif phase == "bagnet":
		copyfile('config_bagnet.py', os.path.join(config['out_dir'],config['date'],'config_bagnet.py'))
	elif not phase is None and 'coarse_to_fine' in phase and "bagnet" in phase:
		copyfile('config_bagnet_ctf.py', os.path.join(config['out_dir'],config['date'],'config_bagnet_ctf.py'))
	else:
		copyfile('config_test.py', os.path.join(config['out_dir'],config['date'],'config_test.py'))

def coarse_cluster_selection(cluster_centers):
	l2 = float("Inf")
	selection = 0
	for i in range(cluster_centers.shape[0]):
		if np.sum(cluster_centers[i]**2) < l2:
			selection = i
			l2 = np.sum(cluster_centers[i]**2)
	return selection

def coarse_image_extraction(sample):
	locations = sample['loations']
	height_interval = sample['image_ori'].shape[0]/(sample['image'].shape[0]/16)
	width_interval = sample['image_ori'].shape[1]/(sample['image'].shape[1]/16)
	output = {}
	for i,location in enumerate(locations):
		height = height_interval*location[0]
		width = width_interval*location[1]
		output[i] = sample['image_ori'][height:height+height_interval,width:width+width_interval]
	return output

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

def save_ckpt(model, optimizer, epoch, loss_tr, loss_cv, config, M,s):
	def do_save(filepath):
		torch.save({
			'epoch': epoch,
			'name': config['model_name'],
			'date': config['date'],
			'learning_rate': config['learning_rate'],
			'model': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			'M':M,
			's':s
			}, filepath)
	# check if best_checkpoint
	is_best_ckpt(epoch,loss_tr,loss_cv,config)
	filepath=os.path.join('..','checkpoint',config['date']+'.pkl')
	do_save(filepath)


def _extract_state_from_dataparallel(checkpoint_dict):
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in checkpoint_dict.items():
		if k.startswith('module.'):
			name = k[7:]
		else:
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
	M = checkpoint['M']
	s = checkpoint['s']
	
	if model:
		
		model_dict = model.state_dict()
		pretrain_dict = checkpoint['model']
		detect_dict = {k: v for k,v in pretrain_dict.items() if k in model_dict}
		full_dict = {}
		#print(list(model_dict.items())[0][0],list(pretrain_dict.items())[0][0])
		
		for k, v in model_dict.items():
			for k_, v_ in pretrain_dict.items():
				if k == k_ or k == k_.split('module.')[-1]:
					full_dict[k] = v_
					print(k,'FOUND')
			if k not in full_dict:
				full_dict[k] = v
				print(k,'NOT FOUND')
		
		model.load_state_dict(_extract_state_from_dataparallel(full_dict))
	return epoch, learning_rate, optimizer, M, s

def K_MEANS(vectors,path_all,config,phase,_bagnet_filter=0.9,_bagnet_batch=100000):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	s={}
	if 'bagnet' in phase:
		size = vectors.shape[1:3]
		vectors_ori = vectors.reshape(-1,vectors.shape[3])
		vectors = vectors_ori[np.random.random(vectors_ori.shape[0])>_bagnet_filter,:] # this is what to go through kmeans, then calculate the assignment for the whole set
	kmeans = KMeans(n_clusters=config['n_cluster'],max_iter=50).fit(vectors)
	M = kmeans.cluster_centers_.transpose(1,0)#dim,n_cluster
	if 'bagnet' in phase:
		# MemoryError! Needs to do the assignment in batch-wise
		assignment = []
		num_batch = vectors_ori.shape[0]//_bagnet_batch
		for i in range(num_batch):
			assignment.append(np.argmin(np.sum((vectors_ori[int(i*_bagnet_batch):int((i+1)*_bagnet_batch),:,np.newaxis]-M[np.newaxis,:,:])**2,axis=1),axis=1))
		assignment.append(np.argmin(np.sum((vectors_ori[int(num_batch*_bagnet_batch):,:,np.newaxis]-M[np.newaxis,:,:])**2,axis=1),axis=1))
		assignment = np.concatenate(assignment,axis=0)
		assignment = assignment.reshape(-1,size[0],size[1])
	else:
		assignment = kmeans.labels_
	for i in range(assignment.shape[0]):
		s[path_all[i]] = assignment[i] # if bagnet, is a 2-dim array
	for i in range(config['n_cluster']):
		print(np.sum(assignment == i)/vectors.shape[0])
	M = torch.tensor(M,requires_grad = False).to(device)#(dim,n_cluster)
	
	return M,s

def point_assign(output,M,_bagnet_batch=100000):
	output = output.reshape(-1,output.shape[-1])
	num_batch = output.shape[0]//_bagnet_batch
	assignment_total = []
	for i in range(num_batch):
		_,assignment = torch.min(torch.sum((output[int(i*_bagnet_batch):int((i+1)*_bagnet_batch),:].cpu().unsqueeze(2)-M.detach().cpu().unsqueeze(0))**2,dim=1),dim=1)
		assignment_total.append(assignment)
	_,assignment = torch.min(torch.sum((output[int(num_batch*_bagnet_batch):,:].cpu().unsqueeze(2)-M.detach().cpu().unsqueeze(0))**2,dim=1),dim=1)
	assignment_total.append(assignment)
	assignment_total = torch.cat(assignment_total,dim=0)
	return assignment_total
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

def centroid_adjustment(x_ori,M,_argmin_ori, config):
	x = x_ori.clone().detach()
	_argmin = _argmin_ori.clone().detach()
	for ind in range(M.shape[-1]):#n_cluster
		mask = (_argmin == ind).byte()
		if torch.any(mask):
			cluster_modify = x[mask,:]#(n,dim)
			cluster_modify = torch.mean(cluster_modify,dim=0)
			M[:,ind] = M[:,ind] - config['cluster_update_multi']*(M[0,:,ind] - cluster_modify)
	return M

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

def loss_para_selection(config,epoch):
	if epoch < config['loss_multi_epoch_switch']:
		return config['ae_loss_multi_ori'],config['cluster_loss_multi_ori']
	else:
		return config['ae_loss_multi_new'], config['cluster_loss_multi_new']

def select_centroid_batch(s,img_name):
	s_batch = np.array([s[i] for i in img_name]).astype(np.int64)
	return torch.tensor(s_batch,requires_grad=False)

def M_new_assignment(_mean,_std,config):
	
	#return torch.cat([_mean[i]+_std[i]*torch.randn(1) for i in range(_mean.shape[0])],dim=0)
	return torch.cat([_mean[i]+_std[i]*(2*config['M_new_assign_rand_multi']*(torch.rand(1)-0.5)) for i in range(_mean.shape[0])],dim=0)

def M_update(M,new_assign,output,config):
	for i in range(M.shape[1]):
		new_mean = torch.mean(output.reshape(-1,output.shape[-1])[new_assign == i],dim=0)
		M[:,i] = M[:,i] - config['cluster_update_multi']*(M[:,i]-new_mean)
	return M[:]

def largest_num_cluster_status(outputs,new_assign,config):
	# Largest number of sample's centroid
	_max_ind = 0
	_count = 0
	for i in range(config['n_cluster']):
		_count_temp = torch.sum(new_assign == i)
		if _count_temp > _count:
			_count = _count_temp
			_max_ind = i
	_mean = torch.mean(outputs[new_assign == _max_ind],dim=0)
	_std = torch.std(outputs[new_assign == _max_ind],dim=0)
	return _mean,_std	

def largest_std_cluster_status(outputs, new_assign,config):
	# Largest std centroid
	_max_ind = 0
	_std_sum = 0
	_mean = None
	_std = None
	for i in range(config['n_cluster']):
		_count_temp = torch.sum(new_assign == i)
		if _count_temp > 10:
			_std_temp = torch.sum(torch.std(outputs[new_assign == i],dim=0))
			if _std_temp > _std_sum:
				_std_sum = _std_temp
				_max_ind = i
				_mean = torch.mean(outputs[new_assign == i],dim=0)
				_std = torch.std(outputs[new_assign == i],dim=0)
	return _mean, _std

def T_SNE(data,assignment,config, phase):
	unique = set(assignment.tolist())
	color_dict = {}
	for ind,i in enumerate(list(unique)):
		color_dict[i] = np.random.rand(3,)
	
	out_dir = config['out_dir']
	name = config['date']
	data = data.reshape(-1,data.shape[-1])
	tsne = manifold.TSNE(n_components=2,init='pca',random_state=0)
	X_tsne = tsne.fit_transform(data)
	for i in range(X_tsne.shape[0]):
		plt.plot(X_tsne[i,0],X_tsne[i,1],'o',color = color_dict[assignment[i]])
	if not os.path.exists(os.path.join(out_dir,name)):
		os.mkdir(os.path.join(out_dir,name))
	plt.savefig(os.path.join(out_dir,name,'TSNE.png'))

def coarse_to_fine_cluster_enlarge(config,epoch):
	extra_cluster_num = 0
	if epoch > config['coarse_to_fine_epoch_start'] and epoch % config['coarse_to_fine_epoch_step'] == 0 and config['n_cluster'] < config['coarse_to_fine_epoch_max']:
		extra_cluster_num = config['coarse_to_fine_n_cluster_step']
		#config['n_cluster'] = config['n_cluster'] + config['coarse_to_fine_n_cluster_step']
		#M = torch.cat((M,torch.zeros(M.shape[0],config['coarse_to_fine_n_cluster_step'])),dim=1)
	return extra_cluster_num

def coarse_to_fine_cluster_enlarge_OneTime(config,M,epoch):
	if epoch > config['coarse_to_fine_epoch_start'] and epoch % config['coarse_to_fine_epoch_step'] == 0 and config['n_cluster'] < config['coarse_to_fine_epoch_max']:
		config['n_cluster'] = config['n_cluster'] + config['coarse_to_fine_n_cluster_step']
		M = torch.cat((M,torch.zeros(M.shape[0],config['coarse_to_fine_n_cluster_step'])),dim=1)
	return  config, M





def test_save_info(epoch,assignment,outputs,config):
	out_dir = config['out_dir']
	epoch = str(epoch)
	name = config['date']
	_std = np.ones(config['n_cluster'])*-1
	for i in range(config['n_cluster']):
		if np.sum(assignment==i) >= 10:		
			_std_temp = np.sum(np.std(outputs[assignment == i],axis=0))
			_std[i] = _std_temp
	
	if not os.path.exists(os.path.join(out_dir,name)):
		os.mkdir(os.path.join(out_dir,name))
	if not os.path.exists(os.path.join(out_dir,name,epoch)):
		os.mkdir(os.path.join(out_dir,name,epoch))
	txt = open(os.path.join(out_dir,name,epoch,'INFO.txt'),'a')
	for i in range(config['n_cluster']):
		txt.write(str(i) + ':\t'+str(_std[i])+'\n')
	txt.close()

