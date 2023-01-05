import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import glob
from skimage.io import imread 
from skimage.transform import resize
import numpy as np
import time 
from PIL import Image
import os
import json

class_representation_dict_2 = {
	0:0,
	1:0,
	2:1
}

class slide_level_classification_dataset(Dataset):
	def __init__(self,config,dataset,phase,transform=None): 
		self.config = config
		self.phase = phase
		self.transform = transform
		self.dataset = dataset
		with open(config['class_path']) as a:
			self.class_dict = json.load(a)
		with open(config['train_val_test_split_path']) as a:
			self.slide_include = json.load(a)[phase+'_slide']
		if dataset == 'UM':
			if config['n_class'] == 2:
				self.class_representation_dict = class_representation_dict_2
			elif config['n_class'] == 3:
				self.class_representation_dict = {0:0,1:1,2:2}
		elif dataset == 'Cervical':
			self.class_representation_dict={}
			for i in range(config['n_class']):
				self.class_representation_dict[int(i)] = int(i)
		self.path_list, self.label = self.path_list_gene()
			
		print('num of data ('+self.phase+'): ',len(self.path_list))
	def __len__(self):
		return len(self.path_list)
	def __getitem__(self,idx):
		image_path = self.path_list[idx]
		image_label = self.label[idx]
		#image = imread(image_path)
		image = np.array(Image.open(image_path))
		#_mean = np.array([0.693, 0.423, 0.535]).reshape(1,-1)
		#_std = np.array([0.099, 0.126, 0.121]).reshape(1,-1)
		#image = (((image/255.)-_mean)/_std).astype(np.float32)
		sample = {'image':image,'label':image_label,'image_path':image_path}
		if not self.transform is None:
			sample = self.transform(sample)
		sample['image'] = torch.tensor(sample['image'].transpose(2,0,1)).float()
		return sample
	def path_list_gene(self):
		path_list = []
		label = []
		for slide_id in self.slide_include:
			if self.dataset == 'UM':
				sample_list = glob.glob(os.path.join(self.config['input_dir'],'Slide '+str(slide_id),'*'))
			elif self.dataset == 'Cervical':
				sample_list = glob.glob(os.path.join(self.config['input_dir'],str(slide_id),'*'))
			path_list = path_list + sample_list
			label = label + [int(self.class_representation_dict[int(self.class_dict[slide_id]['class_ind'])])]*len(sample_list)
		return path_list, label
