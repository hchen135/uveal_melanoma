import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from glob import glob
from skimage.io import imread 
from skimage.transform import resize
import numpy as np
import time 
from PIL import Image


class fine_clustering_dataset(Dataset):
	def __init__(self,config,final_test=False): 
		self.config = config
		if final_test:
			self.path_list = glob('/home/hchen135/uveal_melanoma/data/annotated_data/data_128/*')
		else:	
			if 'data_not_train' not in config or config['data_not_train'] == []:
				self.path_list = glob(config['input_dir']+'/Slide */*')
			else:
				self.path_list = []
				data_to_train = [i for i in range(1,101) if i not in config['data_not_train']]
				for i in data_to_train:
					self.path_list = self.path_list + glob(config['input_dir']+'/Slide '+str(i)+'/*')
		print('num of data: ',len(self.path_list))
		
		# Get len info
		# self.fine_height = self.config.fine_height
		# self.fine_width = self.config.fine_width

		# self.img_ori_size = self.sample['image_ori'].shape # assume hxwxc
		# self.num_height = self.img_ori_size[0]/(self.config.coarse_img_size/16)//self.fine_height
		# self.num_width = self.img_ori_size[1]/(self.config.coarse_img_size/16)//self.fine_width
		# self.num_per_location = int(self.num_height*self.num_width)
		# self.len = int(self.num_per_location*len(self.sample['locations']))
	def __len__(self):
		return len(self.path_list)
	def __getitem__(self,idx):
		# #find which part should be segmented
		# location_number = idx//(self.num_per_location)
		# height_number = (self.len-location_number*self.num_per_location)//self.width_number
		# width_number = (self.len-location_number*self.num_per_location)-num_height*self.width_number
		# # segmnet the part
		# location_info = self.sample['location'][location_number]
		# location_height = location_info[0]*(self.img_ori_size[0]/(self.config.coarse_img_size/16))
		# location_width = location_info[0]*(self.img_ori_size[0]/(self.config.coarse_img_size/16))
		# bbox_height = location_height + height_number*self.fine_height
		# bbox_width = location_width + width_number*self.fine_width
		time_start = time.time()
		fine_image_path = self.path_list[idx]
		fine_image = imread(fine_image_path)
		_mean = np.array([0.735, 0.519, 0.598]).reshape(1,-1)
		_std = np.array([0.067, 0.067, 0.063]).reshape(1,-1)
		fine_image = (((fine_image/255.)-_mean)/_std).astype(np.float32)
		
		#fine_image = Image.open(fine_image_path)
		#fine_image = np.load(fine_image_path).astype(np.float32)
		#print(fine_image_path)
		resize_start_time = time.time()
		#fine_image = resize(fine_image,self.config['input_size'])
		#fine_image = fine_image.transpose(2,0,1).astype(np.float32)
		
		fine_image = F.to_tensor(fine_image)
		time_end = time.time()
		#print(np.max(fine_image))
		fine_sample = {'fine_image':fine_image,'fine_image_path':fine_image_path}
		#print("data preparation time: ", time.time()-time_start)
		return fine_sample



