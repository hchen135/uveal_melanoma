import os
import argparse
import time
import warnings
from sklearn.cluster import KMeans
import time

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import pandas as pd 
from skimage.io import imsave 
from shutil import copyfile

from Pytorch_DeepLab_v3_plus.networks.deeplab_resnet_wodecoder import DeepLabv3_plus
from net import ResNet,DenseNet,BagNet
from dataset import *
from util import load_ckpt,test_save_img, T_SNE, point_assign
from loss import *
from glob import glob
import json
import importlib 

warnings.filterwarnings("ignore")
print(device)
def main(model_name,config_root,checkpoint_root):
	config_path = glob(os.path.join(config_root,model_name,'config*.py'))[0]	
	phase = 'normal'
	if 'bagnet' in config_path.split('/')[-1]:
		phase = 'bagnet'
	print(os.path.exists(config_path))
	#importlib.import_module(config_path,'config')
	spec = importlib.util.spec_from_file_location("config", config_path)
	foo = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(foo)
	config = foo.config
	config['batch_size'] = 16
	print(config)
	# Dataset
	fine_dataset = fine_clustering_dataset(config,final_test = True)
	# Dataloder
	test_loader = DataLoader(
		fine_dataset,
		shuffle = False,
		batch_size = config['batch_size'],
		num_workers = 32)
	# Model
	start_epoch = 0
	if config['model_name'].startswith('resnet'):
		model = ResNet(config)
	elif config['model_name'].startswith('densenet'):
		model = DenseNet(config)
	elif config['model_name'].startswith('deeplab'):
		cluster_vector_dim = config['cluster_vector_dim']
		model = DeepLabv3_plus(nInputChannels=3, n_classes=3, os=16, cluster_vector_dim = cluster_vector_dim, pretrained=True, _print=True)
	elif config['model_name'].startswith('bagnet'):
		model = BagNet(config=config)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	filepath = os.path.join(checkpoint_root,model_name+'.pkl')
	_,_,_,M,_ = load_ckpt(model,filepath)
	config['n_cluster'] = M.shape[1]
	print('n_cluster',config['n_cluster'])
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)
	model.to(device)
	test(test_loader,model,config,M,phase)
	print("Final Test finished ...")

def test(loader, model, config, M, phase):
	print("TESTING START!")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.eval()
	raw_assignment = []
	test_assignment = {}
	test_output = {}
	results = []
	for i, data in enumerate(loader):
		inputs = data['fine_image'].to(device)
		print(inputs.shape)
		img_name = data['fine_image_path']
		
		z = model(inputs)
		if "bagnet" in phase and i == 0:
			size = z.shape[1:3]
		results.append(z.detach().cpu().numpy())
		assignments = point_assign(z,M)
		raw_assignment.append(assignments.detach().cpu().numpy())
		if 'bagnet' in phase:
			assignments = assignments.reshape(-1,size[0],size[1])
		for j, single_img_name in enumerate(img_name):
			test_assignment[single_img_name] = assignments[j].detach().cpu().numpy()
			test_output[single_img_name] = z[j].detach().cpu().numpy()
	raw_assignment = np.concatenate(raw_assignment,axis=0)
	results = np.concatenate(results,axis=0)
	test_save_img(epoch='final_test',assignment=test_assignment, output = test_output, config=config, phase= phase)	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model-name', dest='model_name')
	parser.add_argument('--config-root',dest='config_root',default='/home/hchen135/uveal_melanoma/data/result/fine_extraction',required=False)
	parser.add_argument('--checkpoint-root',dest='checkpoint_root',default='/home/hchen135/uveal_melanoma/checkpoint',required=False)
	args=parser.parse_args()
	main(args.model_name,args.config_root,args.checkpoint_root)

















