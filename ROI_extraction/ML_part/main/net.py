import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torch.autograd import Variable
from bag_of_local_features_models.bagnets.pytorchnet import bagnet9,bagnet17,bagnet33

class ResNet(nn.Module):
	def __init__(self,config):
		super().__init__()
		builder = getattr(models,config['model_name'])
		print('MODEL NAME: ',config['model_name'])
		self.resnet = builder(pretrained = True)
		if config['fixed_feature']:
			for param in self.resnet.parameters():
				param.requires_grad = False
		if config['model_name'] in ['resnet18', 'resnet34']:
			block_4_channel = 512
		else:
			block_4_channel = 2048
		self.conv = nn.Conv2d(block_4_channel,config['cluster_vector_dim'],1)
	def forward(self,x):
		x = self.resnet.conv1(x)
		x = self.resnet.bn1(x)
		x = self.resnet.relu(x)
		x = self.resnet.maxpool(x)
		x = self.resnet.layer1(x)
		x = self.resnet.layer2(x)
		x = self.resnet.layer3(x)
		x = self.resnet.layer4(x)
		z = self.resnet.avgpool(x)
		z = self.conv(z).squeeze(3).squeeze(2)
		z_mean = torch.mean(z,1,True)
		z_std = torch.std(z,1).unsqueeze(1)
		z = (z-z_mean)/z_std
		return z

class DenseNet(nn.Module):
	def __init__(self,config):
		super().__init__()
		builder = getattr(models,config['model_name'])
		print('MODEL NAME: ',config['model_name'])
		self.densenet = builder(pretrained = True)
		if config['fixed_feature']:
			for param in self.densenet.parameters():
				param.requires_grad = False
		if config['model_name'] == 'densenet121':
			block_4_channel = 1024
		elif config['model_name'] == 'densenet169':
			block_4_channel = 1664
		elif config['model_name'] == 'densenet201':
			block_4_channel = 1920
		elif config['model_name'] == 'densenet161':
			block_4_channel = 2208
		self.conv = nn.Conv2d(block_4_channel,config['cluster_vector_dim'],1)
		
	def forward(self,x):
		x = self.densenet.conv1(x)
		x = self.densenet.bn1(x)
		x = self.densenet.relu(x)
		x = self.densenet.maxpool(x)
		x = self.densenet.layer1(x)
		x = self.densenet.layer2(x)
		x = self.densenet.layer3(x)
		x = self.densenet.layer4(x)
		z = self.densenet.avgpool(x)
		z = self.conv(z).squeeze(3).squeeze(2)
		return z

class BagNet(nn.Module):
	def __init__(self,config):
		super().__init__()
		builder = globals()[config['model_name']]
		print('MODEL NAME: ',config['model_name'])
		self.bagnet = builder(pretrained = True)
		if config['fixed_feature']:
			for param in self.densenet.parameters():
				param.requires_grad = False
		block_4_channel = 2048
		self.conv = nn.Conv2d(block_4_channel,config['cluster_vector_dim'],1)
		
		#self.conv1 = nn.Conv2d(3,64,3,2,1)
		#self.maxpool = nn.MaxPool2d(2,2)
		if config['model_name'] == "bagnet17":
			self.avgpool = nn.AvgPool2d(6,4)
		elif config['model_name'] == "bagnet33":
			self.avgpool = nn.AvgPool2d(4,4)
	def forward(self,x):
		#x = self.conv1(x)
		x = self.bagnet.conv1(x)
		x = self.bagnet.conv2(x)
		x = self.bagnet.bn1(x)
		x = self.bagnet.relu(x)
		#x = self.maxpool(x)
		
		x = self.bagnet.layer1(x)
		x = self.bagnet.layer2(x)
		x = self.bagnet.layer3(x)
		x = self.bagnet.layer4(x)
		
		x = self.avgpool(x)
		#x = nn.AvgPool2d(x.size()[2], stride=1)(x)	
		x = self.conv(x)
		x_mean = torch.mean(x,1,True)
		x_std = torch.std(x,1).unsqueeze(1)
		x = (x-x_mean)/x_std
		x = x.permute(0,2,3,1)	
		return x
		

class Net(nn.Module):
	def __init__(self,config):
		super().__init__()
		self.DCN = ResNet(config)
		self.fc = nn.Linear(config['DCN_out_channel'],config['cluster_vector_dim'])

	def forward(self,x):
		x = self.DCN(x)
		x = nn.AdaptiveAvgPool2d((1,1))(x)

		x = x.squeeze(2).squeeze(2)
		x = self.fc(x)
		return x


class Simple_AE(nn.Module):
	def __init__(self, config, fc_hidden1=1024, drop_p=0.3):
		super().__init__()
		self.config = config

		CNN_embed_dim = config['cluster_vector_dim']
		self.fc_hidden1, self.CNN_embed_dim = fc_hidden1, CNN_embed_dim
		# CNN architechtures
		self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
		self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
		self.s1, self.s2, self.s3, self.s4 = (1, 1), (1, 1), (1, 1), (1, 1)      # 2d strides
		self.pd1, self.pd2, self.pd3, self.pd4 = (2, 2), (1, 1), (1, 1), (1, 1)  # 2d padding
	
		self.conv1_e = nn.Conv2d(3,self.ch1,self.k1,self.s1,self.pd1)
		self.batch1_e = nn.BatchNorm2d(self.ch1)
		self.conv2_e = nn.Conv2d(self.ch1,self.ch2,self.k2,self.s2,self.pd2)
		self.batch2_e = nn.BatchNorm2d(self.ch2)
		self.conv3_e = nn.Conv2d(self.ch2,self.ch3,self.k3,self.s3,self.pd3)
		self.batch3_e = nn.BatchNorm2d(self.ch3)
		self.conv4_e = nn.Conv2d(self.ch3,self.ch4,self.k4,self.s4,self.pd4)
		self.batch4_e = nn.BatchNorm2d(self.ch4)
		self.fc1_e = nn.Linear(self.ch4*self.config['input_size'][0]//16*self.config['input_size'][1]//16,self.fc_hidden1)
		self.bn1_e = nn.BatchNorm1d(self.fc_hidden1)
		self.fc2_e = nn.Linear(self.fc_hidden1,self.CNN_embed_dim)
	
		self.fc2_d = nn.Linear(self.CNN_embed_dim,self.fc_hidden1)
		self.bn1_d = nn.BatchNorm1d(self.fc_hidden1)
		self.fc1_d = nn.Linear(self.fc_hidden1,self.ch4*self.config['input_size'][0]//16*self.config['input_size'][1]//16)
		self.batch4_d = nn.BatchNorm2d(self.ch4)
		self.conv4_d = nn.ConvTranspose2d(self.ch4,self.ch3,self.k4,self.s4,self.pd4)
		self.batch3_d = nn.BatchNorm2d(self.ch3)
		self.conv3_d = nn.ConvTranspose2d(self.ch3,self.ch2,self.k3,self.s3,self.pd3)
		self.batch2_d = nn.BatchNorm2d(self.ch2)
		self.conv2_d = nn.ConvTranspose2d(self.ch2,self.ch1,self.k2,self.s2,self.pd2)
		self.batch1_d = nn.BatchNorm2d(self.ch1)
		self.conv1_d = nn.ConvTranspose2d(self.ch1,3,self.k1,self.s1,self.pd1)
	
	def encode(self,x):
		x = self.conv1_e(x)
		x = self.batch1_e(nn.ReLU()(x))
		x = nn.MaxPool2d(2,2)(x)
		x = self.conv2_e(x)
		x = self.batch2_e(nn.ReLU()(x))
		x = nn.MaxPool2d(2,2)(x)
		x = self.conv3_e(x)
		x = self.batch3_e(nn.ReLU()(x))
		x = nn.MaxPool2d(2,2)(x)
		x = self.conv4_e(x)
		x = self.batch4_e(nn.ReLU()(x))
		x = nn.MaxPool2d(2,2)(x)
		x = x.view(x.size(0),-1)
		x = self.fc1_e(x)
		x = self.bn1_e(nn.ReLU()(x))
		x = self.fc2_e(x)
		return x
	def decode(self,x):
		x = self.fc2_d(x)
		x = nn.ReLU()(self.bn1_d(x))
		x = self.fc1_d(x)
		x = x.view(x.size(0), self.ch4, self.config['input_size'][0]//16, self.config['input_size'][1]//16)
		x = nn.ReLU()(self.batch4_d(x))
		x = self.conv4_d(x)
		x = F.interpolate(x, size=(self.config['input_size'][0]//8,self.config['input_size'][1]//8), mode='bilinear')
		x = nn.ReLU()(self.batch3_d(x))
		x = self.conv3_d(x)
		x = F.interpolate(x, size=(self.config['input_size'][0]//4,self.config['input_size'][1]//4), mode='bilinear')
		x = nn.ReLU()(self.batch2_d(x))
		x = self.conv2_d(x)
		x = F.interpolate(x, size=(self.config['input_size'][0]//2,self.config['input_size'][1]//2), mode='bilinear')
		x = nn.ReLU()(self.batch1_d(x))
		x = self.conv1_d(x)
		x = F.interpolate(x, size=(self.config['input_size'][0],self.config['input_size'][1]), mode='bilinear')
		x = nn.Sigmoid()(x)
		return x
		
	def forward(self,x):
		z = self.encode(x)
		x_recon = self.decode(z)
		return x_recon,z

class ResNet_VAE(nn.Module):
    def __init__(self, config, fc_hidden1=1024, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=256):
        super(ResNet_VAE, self).__init__()

        self.config = config

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (7, 7), (7, 7), (7, 7)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = getattr(models,config['model_name'])(pretrained=True)
        modules = list(resnet.children())[:-2]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features*self.config['input_size'][0]//32*self.config['input_size'][1]//32, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * self.config['input_size'][0]//32 * self.config['input_size'][1]//32)
        self.fc_bn5 = nn.BatchNorm1d(64 * self.config['input_size'][0]//32 * self.config['input_size'][1]//32)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )


    def encode(self, x):
        x = self.resnet(x)  # ResNet
        # x = torch.nn.AdaptiveAvgPool2d((1,1))(x) # because MARCC torchvision version is 0.2.1, the average pooling layer is wrong.
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        #x = self.fc1(x)
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        #x = self.fc2(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        #x = self.relu(self.fc4(z))
        #x = self.relu(self.fc5(x)).view(-1, 64, self.config['input_size'][0]//32, self.config['input_size'][1]//32)
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, self.config['input_size'][0]//32, self.config['input_size'][1]//32)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(self.config['input_size'][0],self.config['input_size'][1]), mode='bilinear')
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        #z = self.reparameterize(mu, logvar)
        z = mu
        x_reconst = self.decode(z)

        return x_reconst, z, mu, logvar
