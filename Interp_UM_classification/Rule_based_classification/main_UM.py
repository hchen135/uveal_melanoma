import numpy as np
import pandas as pd 
from BOAmodel_UM import *
from collections import defaultdict
import argparse
import glob
import os
from util import *
from sklearn.linear_model import LogisticRegression
from sklearn import svm

parser = argparse.ArgumentParser()
parser.add_argument('--umap_info_dir', type=str, dest='umap_info_dir',
	help='includes all umap projection information for all sldes.')
parser.add_argument('--method', type=str, dest='method',
	help='logistic or svm or ruleset.')
parser.add_argument('--base_prob', default=0.7, type=float, dest='base_prob',
	help='probability to add points in base maps.')
parser.add_argument('--other_prob', default=0.01, type=float, dest='other_prob',
	help='probability to add points in other maps.')
parser.add_argument('--base_map_num', default=1, type=int, dest='base_map_num',
	help='how many slides are used as base.')
parser.add_argument('--ensemble', action='store_true', dest='ensemble',
	help='how many ensembles to generate.')
parser.add_argument('--ensemble_num', default=100, type=int, dest='ensemble_num',
	help='how many ensembles to generate.')

parser.add_argument('--rho_split', type=str, dest='rho_split',
	help='rho split to split the circle.')
parser.add_argument('--num_theta_split', type=str, dest='num_theta_split',
	help='number of theta split for each rho interval.')
parser.add_argument('--outlier_weight_ratio', type=float, dest='outlier_weight_ratio',
	help='weight ratio for points outside the circle.')


args = parser.parse_args()

def x_y_gene(df):
	return df[df.columns.values[:-2]].astype(float),df.values[:,-1].astype(int)

""" parameters """
# The following parameters are recommended to change depending on the size and complexity of the data
N = 2000      # number of rules to be used in SA_patternbased and also the output of generate_rules
Niteration = 5000  # number of iterations in each chain
Nchain = 2         # number of chains in the simulated annealing search algorithm

supp = 5           # 5% is a generally good number. The higher this supp, the 'larger' a pattern is
maxlen = 2         # maxmum length of a pattern

# \rho = alpha/(alpha+beta). Make sure \rho is close to one when choosing alpha and beta. 
alpha_1 = 500       # alpha_+
beta_1 = 1          # beta_+
alpha_2 = 500         # alpha_-
beta_2 = 1       # beta_-

if __name__ == '__main__':
	args.rho_split = [float(i) for i in args.rho_split.split(':')]
	args.num_theta_split = [float(i) for i in args.num_theta_split.split(':')]

	# first load all data
	all_file_paths = glob(os.path.join(args.umap_info_dir,'umap_proj_*.json'))
	points_dict = load_umap_projection(all_file_paths)
	
	value = []
	rule_num = []
	for i in range(100):
		seed_num = i
		np.random.seed(seed_num)

		df_train, df_valid = BOA_input_data_gene(all_file_paths,points_dict,seed_num,args)

		X_train, y_train = x_y_gene(df_train)
		X_valid, y_valid = x_y_gene(df_valid)

		if args.method in ['logistic', 'svm']:
			X_train = X_train.values
			X_valid = X_valid.values
			if args.method == 'logistic':
				clf = LogisticRegression().fit(X_train, y_train)
			elif args.method == 'svm':
				clf = svm.SVC().fit(X_train, y_train)
			value.append(clf.score(X_valid, y_valid))

		elif args.method == 'ruleset':
			model = BOA(X_train,y_train)
			model.generate_rules(supp,maxlen,N)
			model.set_parameters(alpha_1,beta_1,alpha_2,beta_2,None,None)
			rules = model.SA_patternbased(Niteration,Nchain,print_message=True)

			# test
			Yhat = model.predict(rules,X_valid)
			TP,FP,TN,FN = getConfusion(Yhat,y_valid)
			tpr = float(TP)/(TP+FN)
			fpr = float(FP)/(FP+TN)
			#print ('TP = {}, FP = {}, TN = {}, FN = {} \n accuracy = {}, tpr = {}, fpr = {}'.format(TP,FP,TN,FN, float(TP+TN)/(TP+TN+FP+FN),tpr,fpr))
			value.append(float(TP+TN)/(TP+TN+FP+FN))
			rule_num.append(len(rules))
			for rule in rules:
				print(rule)

		elif args.method == 'ann':
			X_train = X_train.values.astype(np.float32)
			X_valid = X_valid.values.astype(np.float32)

			train_dataset = Ann_dataset(X_train,y_train)
			valid_dataset = Ann_dataset(X_valid,y_valid)

			train_loader = DataLoader(train_dataset,shuffle = True,batch_size = 16,num_workers = 2)
			valid_loader = DataLoader(valid_dataset,shuffle = True,batch_size = 16,num_workers = 2)

			model = Ann(X_train.shape[1])
			optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2,weight_decay = 1e-5)

			for _iter in range(100):
				ann_train(train_loader,model,optimizer)
			acc = ann_test(valid_loader,model)
			value.append(acc)
	print('max acc:',max(value),'avg acc:',np.average(value),'std scc:',np.std(value))
	if len(rule_num) > 0:
		print('avg rule num:',np.average(rule_num),'std rule num:',np.std(rule_num))

