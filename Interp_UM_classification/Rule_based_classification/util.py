import argparse
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from glob import glob
import time
import os
import pickle
import json
import numpy as np
from copy import deepcopy
import itertools
import pandas as pd

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def load_umap_projection(file_paths):
	points_dict = {}
	for path in file_paths:
		with open(path) as a:
			content = json.load(a)
		points_dict[path.split('/')[-1].split('umap_proj_')[-1].split('_info.json')[0]] = content['embedding_loc']
	return points_dict

def add_points(points_dict,map_ind,prob):
	selected_points = []
	for ind in map_ind:
		all_points = points_dict[ind]
		random_prob = np.random.random()*prob
		random_ind = np.random.choice(range(len(all_points)),int(len(all_points)*random_prob),replace=False)
		if len(random_ind) > 0:
			points = np.array(all_points)[random_ind].tolist()
			selected_points.extend(points)
	return selected_points


def generate_ensemble_umap(points_dict,slide_num,ensemble_num,class_num,args):
	x = []
	y_true = []
	for ind in range(ensemble_num):
		# first find a base slide ind
		base_map_ind  = np.random.choice(slide_num,args.base_map_num,replace=False)
		other_map_ind = [i for i in slide_num if i not in base_map_ind]
		# add points to the plot
		selected_points_base = add_points(points_dict,base_map_ind,args.base_prob)
		selected_points_other = add_points(points_dict,other_map_ind,args.other_prob)
		# embeddings
		ensemble_embeddings = selected_points_base + selected_points_other
		count = density_gene(ensemble_embeddings,args.rho_split,args.num_theta_split,args.outlier_weight_ratio)
		y_single = class_num

		x.append(count)
		y_true.append(y_single)
	return x, y_true

def cart2pol(x,y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho,phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def density_gene(embeddings,rho_split=[0,0.5,1],num_theta_split=[4,8],outlier_weight_ratio=1):
	x = np.array(embeddings)[:,0]
	y = np.array(embeddings)[:,1]
	rho, phi = cart2pol(x,y)
	# count values within circle
	count = []
	count_selected = []
	for i in range(len(rho_split)-1):
		count += count_selected
		selected_id = (rho >= rho_split[i])*(rho < rho_split[i+1])
		phi_selected = phi[selected_id]
		count_selected,_ = np.histogram(phi_selected, bins=np.arange(num_theta_split[i]+1)*2*np.pi/num_theta_split[i]-np.pi)
		count_selected = count_selected.tolist()
	# add count values beyond circle
	selected_id = rho >= rho_split[-1]
	phi_selected = phi[selected_id]
	rho_selected = rho[selected_id]
	weights = np.exp((rho_selected-1)*outlier_weight_ratio)
	extra_count,_ = np.histogram(phi_selected, bins=np.arange(num_theta_split[-1]+1)*2*np.pi/num_theta_split[-1]-np.pi, weights=weights)
	assert len(extra_count) == len(count_selected), str(len(extra_count)) + ' ,' + str(len(count_selected))
	count_selected = [count_selected[i]+extra_count[i] for i in range(len(count_selected))]
	count += count_selected

	return np.array(count)/x.shape[0]

def y_true_gene(path):#here path may be int
	if type(path) == int:
		if path <= 50:
			return 0
		else:
			return 1
	elif type(path) == str:
		if int(path.split('/')[-1].split('umap_proj_')[-1].split('_info.json')[0]) <= 50:
			return 0
		else:
			return 1

def data_preparation(points_dict,slide_num,args):
	y_true = []
	x = []
	for num in slide_num:
		embeddings = points_dict[str(num)]
		count = density_gene(embeddings,args.rho_split,args.num_theta_split,args.outlier_weight_ratio)
		if len(embeddings) >0:
			y_single = y_true_gene(num)
			y_true.append(y_single)
			#count = np.insert(count,count.shape[0],num)
			x.append(count)
	return np.array(x), y_true


def input_data_gene(x,y):
	column_names = ['r'+str(i) for i in range(1,x.shape[1]+1)]
	column_names_ratio_part = [str(i)+'/'+str(j) for (i,j) in itertools.combinations(column_names,2)]

	ratio_column_data = []
	for (i,j) in itertools.combinations(range(x.shape[1]),2):
		ratio_column_data.append((x[:,i]/(x[:,j]+1e-7)).tolist())
	ratio_column_data = np.array(ratio_column_data).T
	x_all = np.concatenate([x,ratio_column_data],axis=1)

	x_all = np.concatenate([x_all,np.array(y).reshape(-1,1)],axis=1)

	df_x = pd.DataFrame(x_all,columns = column_names+column_names_ratio_part+["y"])

	return df_x 
def train_val_split(all_file_paths,train_prob=0.8):
	slide_num = [int(path.split('/')[-1].split('umap_proj_')[-1].split('_info.json')[0]) for path in all_file_paths]
	class_ind = [y_true_gene(path) for path in all_file_paths]
	class_1_slide_num = [num for ind,num in enumerate(slide_num) if class_ind[ind] == 0]
	class_2_slide_num = [num for ind,num in enumerate(slide_num) if class_ind[ind] == 1]
	train_class_1_slide_num = np.random.choice(class_1_slide_num,int(len(class_1_slide_num)*train_prob),replace=False).tolist()
	train_class_2_slide_num = np.random.choice(class_2_slide_num,int(len(class_2_slide_num)*train_prob),replace=False).tolist()
	valid_class_1_slide_num = [num for num in class_1_slide_num if num not in train_class_1_slide_num]
	valid_class_2_slide_num = [num for num in class_2_slide_num if num not in train_class_2_slide_num]
	return train_class_1_slide_num, train_class_2_slide_num, valid_class_1_slide_num, valid_class_2_slide_num

def BOA_input_data_gene(all_file_paths,points_dict,seed_num,args):
	np.random.seed(seed_num)
	# train_validation split
	train_class_1_slide_num, train_class_2_slide_num, valid_class_1_slide_num, valid_class_2_slide_num = train_val_split(all_file_paths)
	# generate regular umap values
	X_train, y_train = data_preparation(points_dict,train_class_1_slide_num + train_class_2_slide_num, args)
	X_valid, y_valid = data_preparation(points_dict,valid_class_1_slide_num + valid_class_2_slide_num, args)
	# if ensemble, generate ensemble data
	if args.ensemble:
		class_1_ensemble_num = args.ensemble_num//2
		class_2_ensemble_num = args.ensemble_num//2
		X_train_ensemble_class_1, y_train_ensemble_class_1 = generate_ensemble_umap(points_dict,train_class_1_slide_num,class_1_ensemble_num,0,args)
		X_train_ensemble_class_2, y_train_ensemble_class_2 = generate_ensemble_umap(points_dict,train_class_2_slide_num,class_2_ensemble_num,1,args)
		X_train = np.concatenate([X_train,X_train_ensemble_class_1,X_train_ensemble_class_2],axis=0)
		y_train = np.concatenate([y_train,y_train_ensemble_class_1,y_train_ensemble_class_2],axis=0)
	df_train = input_data_gene(X_train,y_train)
	df_valid = input_data_gene(X_valid,y_valid)
	return df_train, df_valid

class Ann(nn.Module):
    def __init__(self,X_dim,inner_dim=8):
        super(Ann, self).__init__()
        self.Linear_1 = nn.Linear(X_dim, inner_dim)
        self.Linear_2 = nn.Linear(inner_dim,1)

    def forward(self, x):
        x = self.Linear_1(x)
        x = nn.ReLU()(x)
        x = self.Linear_2(x)
        return x
class Ann_dataset(Dataset):
	def __init__(self,input_data,output_data):
		self.X = input_data
		self.y = output_data
	def __len__(self):
		return self.X.shape[0]
	def __getitem__(self, idx):
		return {'X':self.X[idx],'y':self.y[idx]}

def ann_train(loader,model,optimizer):
	losses = []
	model.train()
	loss_fn = torch.nn.BCEWithLogitsLoss()
	for data in loader:
		X = deepcopy(data['X'])
		y = deepcopy(data['y'])

		optimizer.zero_grad()
		z = model(X)
		loss = loss_fn(z,y.float().reshape(-1,1))
		loss.backward()
		optimizer.step()

		del X
		del y
		del z
		

def ann_test(loader,model):
	model.eval()
	out = []
	gt = []
	for data in loader:
		X = data['X']
		y = data['y']

		z = model(X)
		out.extend(z.detach().numpy().reshape(-1).tolist())
		gt.extend(y.detach().numpy().reshape(-1).tolist())

	out = np.array(out) >=0.5
	gt = np.array(gt)
	return np.average(out == gt)




