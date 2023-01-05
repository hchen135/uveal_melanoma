import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import time
import warnings
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import pandas as pd 
from skimage.io import imsave 
from sklearn.metrics import roc_auc_score
from shutil import copyfile

from net import FirstStageNet, SecondStageNet
from dataset import *
from util import config_selection, store_config, save_ckpt, load_ckpt, record_epoch, slide_level_feature_aggregation, batch_generation, naive_slide_aggregation_auc, heatmap_generation, feature_generation, feature_store, store_heatmap_info
from loss import *
from transform import *
from loss import dice_loss
print(torch.cuda.memory_allocated())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
print(device)
def main(config, dataset):
	# Dataset
	train_dataset = slide_level_classification_dataset(config,dataset,'train',transform = NatureFirstStageAugmentation(mean=config['mean'],std=config['std']))
	val_dataset = slide_level_classification_dataset(config,dataset,'val', transform=BackboneTransform(mean=config['mean'],std=config['std']))
	if dataset == 'UM':
		test_dataset = slide_level_classification_dataset(config,dataset,'test', transform=BackboneTransform(mean=config['mean'],std=config['std']))
	
	class_dict = train_dataset.class_dict
	class_representation_dict = train_dataset.class_representation_dict
	# Dataloder
	train_loader = DataLoader(
		train_dataset,
		shuffle = True,
		batch_size = config['batch_size'],
		num_workers = config['batch_size'])
	val_loader = DataLoader(
		val_dataset,
		shuffle = False,
		batch_size = config['batch_size'],
		num_workers = config['batch_size'])
	if dataset == 'UM':
		test_loader = DataLoader(
			test_dataset,
			shuffle = False,
			batch_size = config['batch_size'],
			num_workers = config['batch_size'])
	# Model
	start_epoch = 0
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = FirstStageNet(config,device)
	model_second_stage = SecondStageNet(config,device)

	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)
		model_second_stage = nn.DataParallel(model_second_stage)
	model.to(device)
	model_second_stage.to(device)
	#Optimizer
	optimizer1 = torch.optim.Adam(model.parameters(), lr = config['learning_rate'],weight_decay = 1e-5)
	optimizer2 = torch.optim.Adam(model_second_stage.parameters(), lr = config['learning_rate_second_stage'],weight_decay = 1e-5)
	scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1,patience = config['switch_learning_rate_interval'])
	scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2,patience = config['switch_learning_rate_interval'])
	# learning rate def
	learning_rate_first_stage = config['learning_rate']
	learning_rate_second_stage = config['learning_rate_second_stage']

	store_config(config)
	if config['if_train_first_stage']:
		for epoch in range(start_epoch+1,config['num_epoch']+1):
			loss_tr = train_first_stage(train_loader,model,optimizer1,epoch,config)
			#model = None
			torch.cuda.empty_cache() 
			if config['if_valid'] and epoch % config['valid_epoch_interval'] == 0:
				with torch.no_grad():
					loss_val, acc_val = valid_first_stage(val_loader,model,epoch,class_representation_dict,class_dict,config,optimizer1.param_groups[0]['lr'],dataset)
					torch.cuda.empty_cache()
					scheduler1.step(-acc_val)
				save_ckpt(model,optimizer1,epoch,loss_tr,-acc_val,config)
			else:
				record_epoch(config,epoch)
	if dataset == 'UM':
		if 'if_test_first_stage' in config and config['if_test_first_stage']:
			if config['if_train_first_stage']:
				load_ckpt(model,os.path.join('..','checkpoint',config['date']+'.pkl'))
			else:
				load_ckpt(model,os.path.join('..','checkpoint',config['first_stage_pretrain_model']+'.pkl'))
			with torch.no_grad():
				test_first_stage(test_loader,model,config,class_dict,dataset,class_representation_dict)
	if config['if_train_second_stage']:
		if config['if_train_first_stage']:
			load_ckpt(model,os.path.join('..','checkpoint',config['date']+'.pkl'))
		else:
			load_ckpt(model,os.path.join('..','checkpoint',config['first_stage_pretrain_model']+'.pkl'))
		torch.cuda.empty_cache() 
		print(torch.cuda.memory_allocated())
		print(torch.cuda.memory_cached())
		
		with torch.no_grad():
			# heatmap generation is in second_stage prep!
			train_w_feature_all, train_pred_all, train_slide_num_all, train_pred_softmax_all = second_stage_prep(train_loader,model,config,'train',dataset)
			val_w_feature_all, val_pred_all, val_slide_num_all, val_pred_softmax_all = second_stage_prep(val_loader,model,config,'val',dataset)
		model = None
		torch.cuda.empty_cache() 
		print(torch.cuda.memory_allocated())
		print(torch.cuda.memory_cached())
		val_slide_level_feature_dict,_ = slide_level_feature_aggregation(val_w_feature_all,val_pred_all,val_slide_num_all,val_pred_softmax_all,phase='val')
		if 'synthetic_validation' in config and config['synthetic_validation']:
			val_slide_level_feature_ensemble_dict, val_slide_level_pred_naiive_ensemble_dict = slide_level_feature_aggregation(val_w_feature_all,val_pred_all,val_slide_num_all,val_pred_softmax_all,fake_num_threshold=config['fake_num_threshold'],phase='train')
			with open(os.path.join(config['out_dir'],config['date'],"first_stage.json"),'w') as a:
				json.dump(val_slide_level_pred_naiive_ensemble_dict,a,indent=4)
		for epoch in range(1,config['num_epoch_second_stage']+1):
			agg_phase = 'train' if config['aggregation_augmentation'] else 'val'
			train_slide_level_feature_dict,_ = slide_level_feature_aggregation(train_w_feature_all,train_pred_all,train_slide_num_all,train_pred_softmax_all,phase=agg_phase)
			loss_tr = train_second_stage(model_second_stage, optimizer2, epoch, dataset, class_dict, class_representation_dict, train_slide_level_feature_dict, config)
			#loss_tr = train_second_stage(train_loader,model,model_second_stage,optimizer2,epoch,class_dict,config)
			if config['if_valid'] and epoch % config['valid_epoch_interval'] == 0:
				with torch.no_grad():
					if 'synthetic_validation' in config and config['synthetic_validation']:
						loss_val = valid_second_stage(model_second_stage,epoch,class_dict,class_representation_dict,val_slide_level_feature_ensemble_dict,config,optimizer2.param_groups[0]['lr'],dataset)
					else:
						loss_val = valid_second_stage(model_second_stage,epoch,class_dict,class_representation_dict,val_slide_level_feature_dict,config,optimizer2.param_groups[0]['lr'],dataset)
					scheduler2.step(loss_val)
			else:
				record_epoch(config,epoch)
	#test(test_loader,model,config)
	print("Training finished ...")

def train_first_stage(loader, model, optimizer, epoch, config):# Use slide level label for tile level classification
	print("--- TRAIN START ---")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	losses = []
	if 'first_stage_loss_fn' in config and config['first_stage_loss_fn'] == 'dice_loss':
		loss_fn = dice_loss(config)
	elif 'first_stage_loss_weight' in config :
		loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(config['first_stage_loss_weight']).to(device))
	else:
		loss_fn = nn.CrossEntropyLoss()
	
	model.train()
	date = config['date']
	train_all_time1 = time.time()

	pred_all = []
	labels_all = []
	for i, data in enumerate(loader):
		batch_time1 = time.time()

		inputs = data['image'].to(device)
		labels = data['label'].to(device)
		img_name = data['image_path']
	
		optimizer.zero_grad()
		
		forward_time1 = time.time()
		_,y,f,pred = model(inputs)
		forward_time2 = time.time()
		#print('forward time: ',forward_time2 - forward_time1)
		

		loss_forward_time1 = time.time()
		loss = loss_fn(pred,labels)
		losses.append(loss.item())
		#loss = (loss - config['flooding_level']).abs() + config['flooding_level']
		loss_forward_time2 = time.time()
		#print('loss_forward_time1',loss_forward_time2-loss_forward_time1)

		loss_backward_time1 = time.time()
		loss.backward()
		optimizer.step()
		loss_backward_time2 = time.time()
		#print('loss_backward_time: ',loss_backward_time2-loss_backward_time1)
		
		batch_time = time.time() - batch_time1
		train_log = open("../../log/train_"+date+".txt","a")
		train_log.write("epoch: {0:d}, iter: {1:d}, loss: {2:.3f}, time: {3:.3f}\n".format(epoch,i,loss,batch_time))
		train_log.close()

		# store values for auc calculation
		pred = nn.Softmax(dim=1)(pred)
		pred_all = pred_all + pred.detach().cpu().numpy().tolist()
		for j in range(pred.shape[0]):
			#pred_all.append(nn.Softmax()(pred[j]).detach().cpu().numpy().tolist())
			labels_all.append([1 if i == labels[j] else 0 for i in range(config['n_class'])])
	
	train_auc = roc_auc_score(np.array(labels_all).astype(np.uint8),np.array(pred_all))

	train_all_time2 = time.time()
	train_log = open("../../log/train_"+date+".txt","a")
	train_log.write("TRAINING"+"-"*10 + "epoch: {0:d}, loss: {1:.3f}\n".format(epoch,np.average(losses)))
	train_log.write('total time = '+str(train_all_time2-train_all_time1)+'\n')
	
	train_log.write("TRAINING GLOBAL AUC: "+str(round(train_auc,4))+'\n')
	for i in range(config['n_class']):
		train_auc = roc_auc_score((np.where(np.array(labels_all))[1] == i).astype(np.uint8),np.array(pred_all)[:,i])
		train_log.write("Class "+str(i)+' AUC: '+str(round(train_auc,4))+'\n')

	train_log.close()
	return np.average(losses)


def second_stage_prep(loader,model,config,phase,dataset):
	print("--- TRAIN SECOND STAGE PREP START ---")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	model.eval()
	
	w_feature_all = []
	pred_all = []
	pred_softmax_all = []
	slide_num_all = []
	feature_dict = {0:[],1:[]}
	for i, data in enumerate(loader):
		time1 = time.time()
		inputs = data['image'].to(device)
		labels = data['label'].to(device)
		img_name = data['image_path']
	
		feature,heatmap,w_feature,pred = model(inputs)
		pred_sigmoid = nn.Sigmoid()(pred)
		heatmap = heatmap.detach().cpu().numpy()
		feature = feature.detach().cpu().numpy()
		#print('train.py',heatmap.shape,torch.max(heatmap[0,0]),torch.max(heatmap[0,1]))
	
		w_feature_all.append(w_feature.detach().cpu())
		pred_all.append(pred_sigmoid.detach().cpu())
		if dataset =='UM':
			slide_num_all = slide_num_all + [int(name.split('/')[-2].split(' ')[-1]) for name in img_name]
		elif dataset =='Cervical':
			slide_num_all = slide_num_all + [name.split('/')[-2] for name in img_name]
	
		#store feature and heatmap
		if config['generate_feature']:
			feature_dict = feature_generation(feature_dict,heatmap,feature,labels.detach().cpu().numpy())
		if config['generate_heatmap']:
			heatmap_generation(heatmap,img_name,config)
		time2 = time.time()
		print('one batch time: ',str(round(time2-time1,3)))
		pred_softmax_all.append(nn.Softmax(dim=1)(pred).detach().cpu())
	
	if config['generate_feature']:
		feature_store(feature_dict,config,phase)

	w_feature_all = torch.cat(w_feature_all,dim=0)
	pred_all = torch.cat(pred_all,dim=0)
	pred_softmax_all = torch.cat(pred_softmax_all,dim=0)
	slide_num_all = np.array(slide_num_all).reshape(-1)

	return w_feature_all, pred_all, slide_num_all, pred_softmax_all

	#slide_level_feature_dict = slide_level_feature_aggregation(w_feature_all,pred_all,slide_num_all,phase=phase)
	#return slide_level_feature_dict

def train_second_stage(model_second_stage, optimizer, epoch, dataset, class_dict, class_representation_dict, slide_level_feature_dict, config):
	print("--- TRAIN SECOND STAGE START ---")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	losses = []
	if 'second_stage_loss_fn' in config and config['second_stage_loss_fn'] == 'dice_loss':
		loss_fn = dice_loss(config)
	elif 'second_stage_loss_weight' in config:
		loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(config['second_stage_loss_weight']).to(device))
	else:
		loss_fn = nn.CrossEntropyLoss()
	
	model_second_stage.train()
	date = config['date']
	train_all_time1 = time.time()

	batch_feature, batch_label, batch_slide_id = batch_generation(slide_level_feature_dict,dataset,class_dict, class_representation_dict,config, shuffle=True)
	print(len(batch_feature))

	for i in range(len(batch_feature)):
		batch_time1 = time.time()
		w_feature = batch_feature[i].to(device)
		#print('train.py w_feature in batch_feature: ',w_feature)
		labels = torch.tensor(batch_label[i]).long().to(device)
		slide_id = batch_slide_id[i]
		#print('train second stage slide_id: ',slide_id)
		#print('train second stage corresponding labels: ',labels)
		
		optimizer.zero_grad()
		
		pred = model_second_stage(w_feature)
	
		loss = loss_fn(pred,labels)
		loss.backward()
		optimizer.step()
		losses.append(loss.item())
		
		batch_time = time.time() - batch_time1
		train_log = open("../../log/train_"+date+".txt","a")
		train_log.write("epoch: {0:d}, iter: {1:d}, loss: {2:.3f}, time: {3:.3f}\n".format(epoch,i,loss,batch_time))
		train_log.close()
	train_all_time2 = time.time()
	train_log = open("../../log/train_"+date+".txt","a")
	train_log.write("TRAINING"+"-"*10 + "epoch: {0:d}, loss: {1:.3f}\n".format(epoch,np.average(losses)))
	train_log.write('total time = '+str(train_all_time2-train_all_time1)+'\n')
	train_log.close()
	return np.average(losses)		
'''
def train_second_stage(loader, model, model_second_stage, optimizer, epoch, class_dict, config, dataset):
	print("--- TRAIN SECOND STAGE START ---")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	losses = []
	loss_fn = nn.CrossEntropyLoss()
	
	model.eval()
	model_second_stage.train()
	date = config['date']
	train_all_time1 = time.time()

	w_feature_all = []
	pred_all = []
	slide_num_all = []
	for i, data in enumerate(loader):
		batch_time1 = time.time()
		
		inputs = data['image'].to(device)
		labels = data['label'].to(device)
		img_name = data['image_path']
		
		_,w_feature,pred = model(inputs)
		pred = nn.Sigmoid()(pred)
		
		w_feature_all.append(w_feature.detach().cpu())
		pred_all.append(pred.detach().cpu())		
		slide_num_all = slide_num_all + [int(name.split('/')[-2].split(' ')[-1]) for name in img_name]
	print('model first stage output:',w_feature.reshape(-1)[0])
	print('model first stage output shape: ',w_feature.shape)

	#then aggregation
	slide_level_feature_dict = slide_level_feature_aggregation(w_feature_all,pred_all,slide_num_all)
	batch_feature, batch_label, batch_slide_id = batch_generation(slide_level_feature_dict,dataset,class_dict, config)
	print(len(batch_feature))
	for i in range(len(batch_feature)):
		batch_time1 = time.time()
		w_feature = batch_feature[i].to(device).detach()		
		print('train.py w_feature in batch_feature: ',w_feature)
		labels = torch.tensor(batch_label[i]).long().to(device)
		slide_id = batch_slide_id[i]		

		optimizer.zero_grad()	
		
		pred = model_second_stage(w_feature)
		
		loss = loss_fn(pred,labels)
		print(loss)
		loss.backward()
		optimizer.step()
		losses.append(loss.item())
		
		batch_time = time.time() - batch_time1
		train_log = open("../../log/train_"+date+".txt","a")
		train_log.write("epoch: {0:d}, iter: {1:d}, loss: {2:.3f}, time: {3:.3f}\n".format(epoch,i,loss,batch_time))
		train_log.close()
	train_all_time2 = time.time()
	train_log = open("../../log/train_"+date+".txt","a")
	train_log.write("TRAINING"+"-"*10 + "epoch: {0:d}, loss: {1:.3f}\n".format(epoch,np.average(losses)))
	train_log.write('total time = '+str(train_all_time2-train_all_time1)+'\n')
	train_log.close()
	return np.average(losses)
'''
def valid_first_stage(loader,model,epoch,class_representation_dict,class_dict,config,learning_rate,dataset):
	print("--- VALIDATION START ---")
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	losses = []
	if 'first_stage_loss_fn' in config and config['second_stage_loss_fn'] == 'dice_loss':
		loss_fn = dice_loss(config)
	else:
		loss_fn = nn.CrossEntropyLoss()
	
	model.eval()
	date = config['date']
	pred_all = []
	labels_all = []
	labels_all_single = []
	slide_ind_all = []
	#centroid modification initialization
	for i, data in enumerate(loader):
		batch_time = time.time()
		inputs = data['image'].to(device)
		labels = data['label'].to(device)
		img_name = data['image_path']
		
		_,_,_,pred = model(inputs)

		loss = loss_fn(pred,labels)

		losses.append(loss.item())
		#print('VALIDATION FIRST STAGE LOSS (batch): ',loss.item())
	
		for j in range(pred.shape[0]):
			pred_all.append(nn.Softmax()(pred[j]).detach().cpu().numpy().tolist())
			labels_all.append([1 if i == labels[j] else 0 for i in range(pred.shape[1])])
			labels_all_single.append(labels[j])
			if dataset == 'UM':
				slide_ind_all.append(img_name[j].split('/')[-2].split(' ')[-1])
			elif dataset == 'Cervical':
				slide_ind_all.append(img_name[j].split('/')[-2])
	
	val_auc = roc_auc_score(np.array(labels_all).astype(np.uint8),np.array(pred_all))
	val_acc = np.average(np.array([pred_all[i][labels_all_single[i]] for i in range(len(pred_all))])>0.5)
	naive_aggregated_auc, naive_aggregated_acc, slide_level_label_all, slide_level_pred_all = naive_slide_aggregation_auc(pred_all,slide_ind_all,class_dict,config)
	# store parameters
	val_log = open("../../log/val_"+date+".txt","a")
	if epoch == 1:
		for i in config:
			val_log.write(str(i)+" = "+str(config[i])+"\n")
	# store statistical values
	val_log.write("VALIDATION" + "-"*10 + "epoch: {0:d}, loss: {1:.3f}, learning rate: {2:.7f}\n".format(epoch,np.average(losses),learning_rate))
	val_log.write("VALIDATION GLOBAL AUC: "+str(round(val_auc,4))+', ACC: '+str(round(val_acc,4))+'\n')
	tile_auc = []
	for i in range(config['n_class']):
		val_auc = roc_auc_score((np.where(np.array(labels_all))[1] == i).astype(np.uint8),np.array(pred_all)[:,i])
		tile_auc.append(val_auc)
		val_log.write("Class "+str(i)+' AUC: '+str(round(val_auc,4))+', slide level aggregated AUC: '+str(round(naive_aggregated_auc[i],4))+', slide level aggregated ACC: '+str(round(naive_aggregated_acc,4))+'\n')
	for slide_id in slide_level_pred_all:
		val_log.write('Slide '+str(slide_id)+' prediction: ')
		for i in range(len(slide_level_pred_all[slide_id])):
			val_log.write(str(round(slide_level_pred_all[slide_id][i],4))+', ')
		val_log.write('GT label: '+str(class_representation_dict[int(class_dict[str(slide_id)]['class_ind'])])+', pred: '+str(round(slide_level_pred_all[slide_id][class_representation_dict[int(class_dict[slide_id]['class_ind'])]],4)))
		val_log.write('\n')
	val_log.write('\n')
	val_log.close()
	
	#save results
	#test_save_cell_segmentation(epoch,segmentation_outputs,config)				
	
	return np.average(losses), naive_aggregated_acc

def valid_second_stage(model_second_stage,epoch,class_dict,class_representation_dict,slide_level_feature_dict,config,learning_rate,dataset):
	print("--- VALIDATION SECOND STAGE START ---")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	losses = []
	if 'second_stage_loss_fn' in config and config['second_stage_loss_fn'] == 'dice_loss':
		loss_fn = dice_loss(config)
	else:
		loss_fn = nn.CrossEntropyLoss()
	
	model_second_stage.eval()
	date = config['date']
	train_all_time1 = time.time()
	
	batch_feature, batch_label, batch_slide_id = batch_generation(slide_level_feature_dict,dataset,class_dict,class_representation_dict,config)
	print(len(batch_feature))

	prediction_all = {}
	for i in range(len(batch_feature)):
		w_feature = batch_feature[i].to(device).detach()
		#print('train.py w_feature in batch_feature IN VALIDATION: ',w_feature)
		labels = torch.tensor(batch_label[i]).long().to(device)
		slide_id = batch_slide_id[i]
		
		pred = model_second_stage(w_feature)

		loss = loss_fn(pred,labels)
		losses.append(loss.item())

		for j in range(len(slide_id)):
			prediction_all[slide_id[j]] = nn.Softmax()(pred[j]).detach().cpu().numpy().astype(np.float32).tolist()
			
	val_log = open("../../log/val_"+date+".txt","a")
	if epoch == 1:
		for i in config:
			val_log.write(str(i)+" = "+str(config[i])+"\n")
	val_log.write("VALIDATION" + "-"*10 + "epoch: {0:d}, loss: {1:.3f}, learning rate: {2:.7f}\n".format(epoch,np.average(losses),learning_rate))
	global_labels = []
	global_preds = []
	for slide_id in prediction_all:
		pred = prediction_all[slide_id]
		if config['n_class'] == 2:
			global_preds.append(pred[1])
			global_labels.append(class_representation_dict[int(class_dict[str(slide_id).split('_')[0]]['class_ind'])])
		else: # multiclass
			global_preds.append(pred)
			global_labels.append(class_representation_dict[int(class_dict[str(slide_id).split('_')[0]]['class_ind'])])
		val_log.write('Slide '+str(slide_id)+' prediction: ')
		for i in range(len(prediction_all[slide_id])):
			val_log.write(str(round(prediction_all[slide_id][i],4))+', ')
		val_log.write('GT label: '+str(class_representation_dict[int(class_dict[str(slide_id).split('_')[0]]['class_ind'])])+', pred: '+str(round(prediction_all[slide_id][class_representation_dict[int(class_dict[str(slide_id).split('_')[0]]['class_ind'])]],4)))
		val_log.write('\n')
	
	if config['n_class'] == 2:
		val_auc = roc_auc_score(np.array(global_labels).astype(np.uint8),np.array(global_preds))
		val_acc = np.average([global_preds[i] > 0.5 if global_labels[i] == 1 else global_preds[i] < 0.5 for i in range(len(global_preds))])
	else:
		val_auc = roc_auc_score(np.array(global_labels).astype(np.uint8),np.array(global_preds),multi_class='ovo')
		val_acc = np.sum(np.argmax(global_preds,axis=1) == np.array(global_labels))/len(global_labels)
	val_log.write("VALIDATION GLOBAL AUC: "+str(round(val_auc,4))+'\n')
	val_log.write("VALIDATION GLOBAL ACC: "+str(round(val_acc,4))+'\n')
	val_log.close()
	
	# save to json file
	with open(os.path.join(config['out_dir'],config['date'],"second_stage_"+str(epoch)+".json"),'w') as a:
		json.dump(prediction_all,a,indent=4)
	return np.average(losses)
		
def test_first_stage(loader, model, config, class_dict,dataset,class_representation_dict):
	print("FIRST STAGE TESTING START!")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.eval()
	date = config['date']

	general_heatmap_all = {}
	detail_heatmap_all = {}
	pred_all = []
	labels_all = []
	labels_all_single = []
	slide_ind_all = []
	for i, data in enumerate(loader):
		inputs = data['image'].to(device)
		labels = data['label'].to(device)
		img_name = data['image_path']

		_,y_sigmoid,_,pred = model(inputs)
		for j in range(pred.shape[0]):
			pred_all.append(nn.Softmax()(pred[j]).detach().cpu().numpy().tolist())
			labels_all.append([1 if k == labels[j] else 0 for k in range(pred.shape[1])])
			labels_all_single.append(labels[j])
			if dataset == 'UM':
				slide_ind_all.append(img_name[j].split('/')[-2].split(' ')[-1])
			elif dataset == 'Cervical':
				slide_ind_all.append(img_name[j].split('/')[-2])
	
		#store the results:
		store_heatmap_info(general_heatmap_all, detail_heatmap_all, y_sigmoid.detach().cpu().numpy(),img_name)
	# write predictions to disk
	naive_aggregated_auc, naive_aggregated_acc, slide_level_label_all, slide_level_pred_all = naive_slide_aggregation_auc(pred_all,slide_ind_all,class_dict,config)
	val_log = open("../../log/val_"+date+".txt","a")
	for slide_id in slide_level_pred_all:
		val_log.write('Slide '+str(slide_id)+' prediction: ')
		for i in range(len(slide_level_pred_all[slide_id])):
			val_log.write(str(round(slide_level_pred_all[slide_id][i],4))+', ')
		val_log.write('GT label: '+str(class_representation_dict[int(class_dict[str(slide_id)]['class_ind'])])+', pred: '+str(round(slide_level_pred_all[slide_id][class_representation_dict[int(class_dict[slide_id]['class_ind'])]],4)))
		val_log.write('\n')
	val_log.write('\n')
	val_log.close()
	# store heatmaps
	with open(os.path.join(config['out_dir'],config['date'],'detail_heatmap_info.json'),'w') as a:
		json.dump(detail_heatmap_all,a,indent=4)
	with open(os.path.join(config['out_dir'],config['date'],'general_heatmap_info.json'),'w') as a:
		json.dump(general_heatmap_all,a,indent=4)
	

def test(loader, model, config):
	print("TESTING START!")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	semantic_segmentation_outputs = {}
	model.eval()
	#centroid modification initialization
	for i, data in enumerate(loader):
		batch_time = time.time()
		inputs = data['image'].to(device)
		SLIC_label = data['SLIC_label'].to(device)
		SLIC_mask = data['SLIC_mask'].numpy()
		SLIC_mask = torch.from_numpy(SLIC_mask).to(device)
		img_name = data['image_path']

		out = model(inputs,SLIC_mask)
		
		out_prob = nn.Sigmoid()(out)
		print(torch.max(out_prob))
		prediction = suppix_semantic_seg_pred_visualization(out_prob.detach().cpu().numpy(),SLIC_mask.detach().cpu().numpy())
		
		for _ind in range(len(img_name)):
			semantic_segmentation_outputs[img_name[_ind]] = prediction[_ind]
	
		print('testing finished:',i)
	test_save_suppix_semantic_segmentation('test',semantic_segmentation_outputs,config)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', dest='dataset',default='UM')
	args=parser.parse_args()
	config = config_selection(args.dataset)
	for i in config:
		print(i,':',config[i])
	main(config, args.dataset)

















