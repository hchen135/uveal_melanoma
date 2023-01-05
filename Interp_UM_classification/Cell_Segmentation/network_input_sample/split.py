from glob import glob
import numpy as np


np.random.seed(1)

img_list = glob('*/*.png')
num_sample = len(img_list)


random = np.random.choice(num_sample,num_sample,replace=False)


def select(random,img_list):
	output = []
	for i in random:
		path = img_list[i]
		output.append('/'.join([path.split('/')[-2],path.split('/')[-1].split('.')[0]]))
	return output

train_num = int(num_sample*8//10)
val_num = int(num_sample*1//10)
test_num = int(num_sample*1//10)

train_list = select(random[:train_num],img_list)
val_list = select(random[train_num:train_num+val_num],img_list)
test_list = select(random[train_num+val_num:],img_list)

def save(list_,phase):
	with open(phase+'.txt','w') as a:
		content = '\n'.join(list_)
		a.write(content)

save(train_list,'train')
save(val_list,'val')
save(test_list,'test')
