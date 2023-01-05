import json
import argparse
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
import time
import umap

# Nature colors
# ['#D55E00', '#CC79A7', '#0072B2', '#F0E442', '#009E73']

def main(exp_name,phase,cluster_method,ratio=0.01):
	np.random.seed(1)
	#load the stored feature
	feature_dict_path = '../heatmap/'+exp_name+'/feature_dict_'+phase+'.json'
	with open(feature_dict_path) as a:
		feature_dict = json.load(a)
	feature_1 = feature_dict['0']
	feature_2 = feature_dict['1']
	slide_label_1 = feature_dict['label_0']
	slide_label_2 = feature_dict['label_1']
	print(len(feature_1[0]))
	#TSNE
	color_dict = [[],[]]
	for i in range(2):
		a = np.random.rand(3)
	color_dict[0] = ['#fec9f7ff','#cc79a7ff']
	color_dict[1] = ['#0072b2ff','#8BE8FFff']
	#color_dict[0] = ['#ffaaaa','#ff0000']# [0]: feature belongs to class 1 and image belongs to class 1, [1]: feature belongs to class 1 and image belongs to class 2
	#color_dict[1] = ['#0066ff','#aaccff']# [0]: feature belongs to class 2 and image belongs to class 1, [1]: feature belongs to class 2 and imaeg belongs to class 2

	legend_labels = [['feature_label 1, slide_label 1','feature_label 1, slide_label 2'],['feature_label 2, slide_label 1','feature_label 2, slide_label 2']]
	
	feature_all = np.array(feature_1 + feature_2)
	print(feature_all.shape)
	label_all = np.array([0]*len(feature_1) + [1]*len(feature_2))
	slide_label_all = np.array(slide_label_1 + slide_label_2)
	time1 = time.time()
	
	random_index = np.random.random(feature_all.shape[0]) < ratio
	print("selected number of features: ",np.sum(random_index))
	feature_partial = feature_all[random_index,:]
	label_partial = label_all[random_index]
	slide_label_partial = slide_label_all[random_index]
	print("pred_label 1, slide_label 1: ",np.sum((label_all == 0)*(slide_label_all == 0)))
	print("pred_label 1, slide_label 2: ",np.sum((label_all == 0)*(slide_label_all == 1)))
	print("pred_label 2, slide_label 1: ",np.sum((label_all == 1)*(slide_label_all == 0)))
	print("pred_label 2, slide_label 2: ",np.sum((label_all == 1)*(slide_label_all == 1)))
	if cluster_method == 'tsne':
		tsne = manifold.TSNE(n_components=2,init='pca',random_state=0)
		X_cluster = tsne.fit_transform(feature_partial)
	elif cluster_method == 'umap':
		reducer = umap.UMAP()
		X_cluster = reducer.fit_transform(feature_partial)
	time2 = time.time()
	print('cluster_time:',round(time2-time1,4))
	for i in range(X_cluster.shape[0]):
		if label_partial[i] == slide_label_partial[i]:
			marker_size = 0.5
		else:
			marker_size = 1.0
		plt.plot(X_cluster[i,0],X_cluster[i,1],'o',color=color_dict[label_partial[i]][slide_label_partial[i]],label = legend_labels[label_partial[i]][slide_label_partial[i]],markersize=marker_size)
	
	legend_elements = []
	for i in range(2):
		for j in range(2):
			legend_elements.append(Line2D([0], [0], marker='o', color=color_dict[i][j], label=legend_labels[i][j], markersize=2))
	#plt.legend(handles=legend_elements)
	plt.axis('off')	

	if cluster_method == 'tsne':
		plt.savefig('../heatmap/'+exp_name+'/feature_visualization_TSNE_'+phase+'.pdf')
	elif cluster_method == 'umap':
		plt.savefig('../heatmap/'+exp_name+'/feature_visualization_UMAP_'+phase+'.pdf')
	time3 = time.time()
	print('Plot time:',round(time3-time2,4))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp-name', dest='exp_name')
	parser.add_argument('--phase', dest='phase')
	parser.add_argument('--cluster-method',dest='cluster_method')
	args=parser.parse_args()
	main(args.exp_name,args.phase,args.cluster_method)


