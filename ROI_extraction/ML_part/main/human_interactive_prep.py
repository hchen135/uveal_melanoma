import numpy as np
import json
import argparse
import torch
import os
from net import *
from util import *
from skimage.io import imread
import torchvision.transforms.functional as F
import importlib.util

def import_config(config_path):
	spec = importlib.util.spec_from_file_location("config", config_path)
	foo = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(foo)
	return foo.config

def preprocessing(img):
	_mean = np.array([0.735, 0.519, 0.598]).reshape(1,-1)
	_std = np.array([0.067, 0.067, 0.063]).reshape(1,-1)
	img = (((img/255.)-_mean)/_std).astype(np.float32)
	img = F.to_tensor(img)
	return img
	
def main(args):
	# path definition
	epoch_store = args.epoch
	print(epoch_store)
	result_path = os.path.join(args.result_dir, args.exp_name+'/'+str(epoch_store)+'/pred.txt')
	checkpoint_path = os.path.join(args.checkpoint_dir, args.exp_name+'.pkl')
	#checkpoint_path = os.path.join(args.checkpoint_dir, '03112020thin6V2_bagnet17_lr-3_epoch20_NoCTF_clusterdim16_batch32_gpu2_ClusterUpdateMulti0.1_NewInitialDIRECT_NewAssignMulti0.5_UniformAss_bagnetBN0.1_KMEANInitial0.1Data_largestSTD_hierarchy.pkl')
	sample_img_path = os.path.join(args.sample_img_dir,args.sample_img_name)
	#load raw data
	with open(result_path,'r') as a:
		result = json.load(a)
	try:
		config = import_config(os.path.join(args.result_dir,args.exp_name,'config_bagnet_ctf.py'))
	except:
		try:
			config = import_config(os.path.join(args.result_dir,args.exp_name,'config_bagnet.py'))
		except:
			config = import_config(os.path.join(args.result_dir,args.exp_name,'config_test.py'))
	model = BagNet(config)
	epoch,learning_rate,optimizer,M,s = load_ckpt(model,checkpoint_path)
	# load and preprocess img
	img = imread(sample_img_path)	
	img = preprocessing(img)
	img = img.unsqueeze(0)
	# get the feature vector as inital point	
	model.eval()
	feature = model(img)
	# avg feature calculation
	avg_feature = nn.AvgPool2d(feature.size()[2], stride=1)(feature.permute(0,3,1,2))
	avg_feature = avg_feature.reshape(-1).detach().numpy().tolist()
	
	# prepare output data
	output = {}
	output['M'] = M.numpy().tolist()
	output['assignment'] = result
	output['human_initial_point'] = avg_feature
	
	if args.phase == 'V2':
		feature_result_path = os.path.join(args.result_dir, args.exp_name+'/'+str(epoch_store)+'/pred_vector.txt')
		with open(feature_result_path,'r') as a:
			feature_result = json.load(a)
		output['feature_vector'] = feature_result
	# store output data
	with open(os.path.join(args.output_dir,args.exp_name+'.txt'),'w') as outfile:
		json.dump(output,outfile,indent=4)
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp-name', dest='exp_name')
	parser.add_argument('--result-dir',dest='result_dir',default='/home/hchen135/uveal_melanoma/data/result/fine_extraction/',required=False)
	parser.add_argument('--checkpoint-dir',dest='checkpoint_dir',default='/home/hchen135/uveal_melanoma/checkpoint',required=False)
	parser.add_argument('--sample-img-dir',dest='sample_img_dir',default='/home/hchen135/uveal_melanoma/data/generated_data/CoarseExtractionV2_128/',required=False)
	parser.add_argument('--sample-img-name',dest='sample_img_name',default='Slide 10/304_TileLoc_16_260.png',required=False)
	parser.add_argument('--output-dir',dest='output_dir',default='/home/hchen135/uveal_melanoma/human_interactive_prep',required=False)
	parser.add_argument('--phase',dest='phase',default='V2',required=False)
	parser.add_argument('--epoch',dest='epoch',default='test',required=False)
	args=parser.parse_args()
	main(args)
