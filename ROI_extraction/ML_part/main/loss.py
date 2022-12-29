import torch.nn as nn
import torch
import torch.nn.functional as F
import time 

class DCN_cluster_loss(nn.Module):
	def __init__(self,config):
		super().__init__()
		self.config = config
	def forward(self,x,M,s):
		s = s.reshape(-1)
		x = x.reshape(-1,x.shape[-1])
		target = M[:,s].transpose(0,1)#(batch,dim), if bagnet, (batch*width*height,dim)
		
		loss = torch.mean((x-target)**2)

		return loss

class VAE_loss(nn.Module):
	def __init__(self,config):
		super().__init__()
		self.config = config
		self.mse_loss = nn.MSELoss()
	def forward(self,recon_x, x):
		#BCE = F.binary_cross_entropy(recon_x.reshape(recon_x.shape[0], -1),
		#			x.view(x.shape[0], -1), size_average=False)
		time1 = time.time()
		diff = (x-recon_x)**2
		time2 = time.time()
		print("DIFF time: ", time2 - time1)
		MSE = torch.mean(diff)
		time3 = time.time()
		print("MEAN time: ", time3 - time2)
		#MSE = self.mse_loss(x,recon_x)
		# see Appendix B from VAE paper:
		# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
		# https://arxiv.org/abs/1312.6114
		# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		#KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		#return BCE + KLD
		#print('vae_loss',MSE)
		return MSE

class loss_all(nn.Module):
	def __init__(self,config):
		super().__init__()
		self.config = config
		self.DCN_cluster_loss = DCN_cluster_loss(self.config)
		#self.VAE_loss = VAE_loss(self.config)
	def forward(self,z,M,s,x=None,recon_x=None):
		#clustering
		cluster_loss = self.DCN_cluster_loss(z,M,s)
		return cluster_loss



