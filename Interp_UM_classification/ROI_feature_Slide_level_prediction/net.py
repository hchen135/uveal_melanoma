import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torch.autograd import Variable

from bag_of_local_features_models.bagnets.pytorchnet import bagnet9,bagnet17,bagnet33
from resnet_modified import *
from DANet import DANetHead

class ResNet(nn.Module):
	def __init__(self,config,device):
		super().__init__()
		self.device = device
		builder = getattr(models,config['model_name'])
		#builder = globals()[config['model_name']]
		print('MODEL NAME: ',config['model_name'])
		self.resnet = builder(pretrained = True)
		if config['fixed_feature']:
			for param in self.resnet.parameters():
				param.requires_grad = False
		if config['model_name'] in ['resnet18', 'resnet34']:
			block_4_channel = 512
		else:
			block_4_channel = 2048
		if 'first_stage_feature_channel' in config:
			feature_channel = config['first_stage_feature_channel']
		else:
			feature_channel = 32
		self.conv1 = nn.Conv2d(block_4_channel,feature_channel,1)
	def forward(self,x):
		x = self.resnet.conv1(x)
		x = self.resnet.bn1(x)
		x = self.resnet.relu(x)
		x = self.resnet.maxpool(x)
		x = self.resnet.layer1(x)
		x = self.resnet.layer2(x)
		x = self.resnet.layer3(x)
		x = self.resnet.layer4(x)
		#x = self.conv1(x)
		#print(x.shape)

		return x

class BagNet(nn.Module):
	def __init__(self,config,device):
		super().__init__()
		self.device = device
		builder = globals()[config['model_name']]
		print('MODEL NAME: ',config['model_name'])
		self.bagnet = builder(pretrained = True)
		if config['fixed_feature']:
			for param in self.resnet.parameters():
				param.requires_grad = False
		block_4_channel = 2048
		if 'first_stage_feature_channel' in config:
			feature_channel = config['first_stage_feature_channel']
		else:
			feature_channel = 32
		self.conv1 = nn.Conv2d(block_4_channel,feature_channel,1)
	def forward(self,x):
		x = self.bagnet.conv1(x)
		x = self.bagnet.conv2(x)
		x = self.bagnet.bn1(x)
		x = self.bagnet.relu(x)

		x = self.bagnet.layer1(x)
		x = self.bagnet.layer2(x)
		x = self.bagnet.layer3(x)
		x = self.bagnet.layer4(x)

		x = self.conv1(x)
		
		return x


class FirstStageNet(nn.Module):
	def __init__(self,config,device):
		super().__init__()
		self.config = config
		if 'resnet' in config['model_name']:
			self.bottleneck = ResNet(config,device)
			downsampled_size = config['img_size']//32
		elif 'bagnet' in config['model_name']:
			self.bottleneck = BagNet(config,device)
			downsampled_size = 28
		if 'first_stage_feature_channel' in config:
			feature_channel = config['first_stage_feature_channel']
		else:
			feature_channel = 32
		self.const = torch.tensor(np.log((downsampled_size)**2)).float().detach()

		try:
			self.LSEPooling_scaling = config['LSEPooling_scaling']
		except:
			self.LSEPooling_scaling = 1

		if 'DANet' in self.config and self.config['DANet']:
			self.DANetHead = DANetHead(2048,config['n_class'],nn.BatchNorm2d,config['DADiv1'],config['DADiv2'])
		else:
			self.conv2 = nn.Conv2d(feature_channel,config['n_class'],1)
	def forward(self,x):
		x = self.bottleneck(x)
		if 'DANet' in self.config and self.config['DANet']:
			x,y = self.DANetHead(x)
		else:
			y = self.conv2(x)# x now could be the heatmap
		y_sigmoid = nn.Sigmoid()(y)
		
		z = y.reshape(y.shape[0],y.shape[1],-1)
		z = (torch.logsumexp(self.LSEPooling_scaling*z,dim=2)-self.const)/self.LSEPooling_scaling#[batch_size,n_class], tile level prediction
		#z = nn.AvgPool2d(y.shape[-1])(y).squeeze(-1).squeeze(-1)		

		# normalzied global feature 
		f = (x.unsqueeze(1)*y_sigmoid.unsqueeze(2)).reshape(x.shape[0],y_sigmoid.shape[1],x.shape[1],-1)#pixel-wise weighted feature
		f = torch.sum(f,dim=3)# aggregated feature
		w = torch.sum(y_sigmoid.reshape(y_sigmoid.shape[0],y_sigmoid.shape[1],-1),dim=2).unsqueeze(2)# normalization value
		f = f/w# normalized feature
		f = f*nn.Sigmoid()(z.unsqueeze(2))# weighted global feature
		print(f.shape)
		'''
		print(torch.mean(f))
		print(torch.mean(x))
		'''
		w = None
		return x,y_sigmoid,f,z

class SecondStageNet(nn.Module):
	def __init__(self,config,device):
		super().__init__()
		self.device = device
		self.config = config
		'''
		if config['model_name'] in ['resnet18', 'resnet34']:
			input_dim = 512
		else:
			input_dim = 2048
		'''
		if 'DANet' in self.config and self.config['DANet']:
			input_dim = 2048 // int(self.config['DADiv1']) // int(self.config['DADiv2'])
		else:
			input_dim = self.config['first_stage_feature_channel']
		hidden_dim = self.config['second_stage_hidden_layer_num']
		output_dim = self.config['n_class']
		for i in range(output_dim):
			setattr(self,'fc1_'+str(i),nn.Linear(input_dim,hidden_dim))
			setattr(self,'bn1_'+str(i),nn.BatchNorm1d(hidden_dim))
			setattr(self,'relu_'+str(i),nn.ReLU())
			setattr(self,'dropout1_'+str(i),nn.Dropout())
			setattr(self,'fc2_'+str(i),nn.Linear(hidden_dim,1))

	def forward(self,w_f):
		for i in range(self.config['n_class']):		
			setattr(self,'x_'+str(i),getattr(self,'fc1_'+str(i))(w_f[:,i,:]))
			setattr(self,'x_'+str(i),getattr(self,'bn1_'+str(i))(getattr(self,'x_'+str(i))))
			setattr(self,'x_'+str(i),getattr(self,'relu_'+str(i))(getattr(self,'x_'+str(i))))
			setattr(self,'x_'+str(i),getattr(self,'dropout1_'+str(i))(getattr(self,'x_'+str(i))))
			setattr(self,'x_'+str(i),getattr(self,'fc2_'+str(i))(getattr(self,'x_'+str(i))))#(batch_size,1)
		print('Second network output shape (single class): ',self.x_0.shape)
		out = [getattr(self,'x_'+str(i)) for i in range(self.config['n_class'])]
		print('Second Stage Output: ',torch.cat(out,dim=1))	
		return torch.cat(out,dim=1)
