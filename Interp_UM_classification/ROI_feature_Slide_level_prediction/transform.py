import numpy as np
from torchvision import transforms
from copy import deepcopy
from skimage.color import rgb2hsv,hsv2rgb
from numpy import random

class Compose(object):
	def __init__(self,transforms):
		self.transforms = transforms

	def __call__(self,data_dict):
		for t in self.transforms:
			data_dict = t(data_dict)
		return data_dict

class ConvertFromInts(object):
	def __call__(self,data_dict):
		data_dict['image'] = data_dict['image'].astype(np.float32)
		return data_dict

class RandomSaturation(object):
	def __init__(self, lower=0.5, upper=1.5):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "contrast upper must be >= lower."
		assert self.lower >= 0, "contrast lower must be non-negative."

	def __call__(self,data_dict):
		if random.randint(2):
			data_dict['image'][:,:,1] *= random.uniform(self.lower, self.upper)

		return data_dict

class RandomHue(object):
	def __init__(self, delta=18.0):
		assert delta >= 0.0 and delta <= 360.0
		self.delta = delta

	def __call__(self,data_dict):
		if random.randint(2):
			data_dict['image'][:,:,0] += random.uniform(-self.delta, self.delta)
			data_dict['image'][:,:,0][data_dict['image'][:,:,0] > 360.0] -= 360.0
			data_dict['image'][:,:,0][data_dict['image'][:,:,0] < 0.0] += 360.0
		return data_dict

class RandomLightingNoise(object):
	def __init__(self):
		 self.perms = ((0, 1, 2), (0, 2, 1),
			       (1, 0, 2), (1, 2, 0),
			       (2, 0, 1), (2, 1, 0))

	def __call__(self,data_dict):
		return data_dict

class RandomContrast(object):
	def __init__(self, lower=0.5, upper=1.5):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "contrast upper must be >= lower."
		assert self.lower >= 0, "contrast lower must be non-negative."

	def __call__(self,data_dict):
		if random.randint(2):
			alpha = random.uniform(self.lower, self.upper)
			data_dict['image'] *= alpha
		return data_dict

class RandomBrightness(object):
	def __init__(self, delta=32):
		assert delta >= 0.0
		assert delta <= 255.0
		self.delta = delta
	
	def __call__(self,data_dict):
		if random.randint(2):
			delta = random.uniform(-self.delta, self.delta)
			data_dict['image'] += delta
		return data_dict

class ConvertColor(object):
	def __init__(self,current='rgb',transform='hsv'):
		self.current = current
		self.transform = transform

	def __call__(self,data_dict):
		if self.current == 'rgb' and self.transform == 'hsv':
			data_dict['image'] = rgb2hsv(data_dict['image'])
		elif self.current == 'hsv' and self.transform == 'rgb':
			data_dict['image'] = hsv2rgb(data_dict['image'])
		else:
			raise NotImplementedError
		return data_dict

class PhotometricDistort(object):
	def __init__(self):
		self.pd = [
			RandomContrast(),
			ConvertColor(transform='hsv'),
			RandomSaturation(),
			RandomHue(),
			ConvertColor(current='hsv', transform='rgb'),
			RandomContrast()
		]
		self.rand_brightness = RandomBrightness()
		self.rand_light_noise = RandomLightingNoise()

	def __call__(self,data_dict):
		data_dict_copy = deepcopy(data_dict)
		data_dict_copy = self.rand_brightness(data_dict_copy)
		
		if random.randint(2):
			distort = Compose(self.pd[:-1])
		else:
			distort = Compose(self.pd[1:])
		data_dict_copy = distort(data_dict_copy)
		return self.rand_light_noise(data_dict_copy)
		

class RandomMirror(object):
	def __call__(self,data_dict):
		data_dict['image'] = data_dict['image'][:,::-1].copy()
		return data_dict
	
class RandomFlip(object):
	def __call__(self,data_dict):
		data_dict['image'] = data_dict['image'][::-1].copy()	
		return data_dict

class RandomRot90(object):
	def __call__(self,data_dict):
		k = np.random.randint(4)
		data_dict['image'] = np.rot90(data_dict['image'],k).copy()
		return data_dict

class BackboneTransform(object):
	def __init__(self, mean, std):
		self.mean = np.array(mean, dtype=np.float32).reshape(1,1,-1)
		self.std  = np.array(std,  dtype=np.float32).reshape(1,1,-1)

	def __call__(self,data_dict):
		
		data_dict['image'] = data_dict['image'].astype(np.float32)
		data_dict['image'] = (data_dict['image'] - self.mean) / self.std
		
		return data_dict

class NatureFirstStageAugmentation(object):
	def __init__(self,mean,std):
		self.augment = Compose([
			ConvertFromInts(),
			PhotometricDistort(),
			RandomMirror(),
			RandomFlip(),
			RandomRot90(),
			BackboneTransform(mean,std)
		])	
	def __call__(self,data_dict):
		return self.augment(data_dict)
