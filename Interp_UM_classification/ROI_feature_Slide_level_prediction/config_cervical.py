config = {
	'date':'04082021_Nature_cervical_4class_ResNet50_DANet_dice0.5_LR2e-4_70_30_LSEPoolScaling1_ImgAug_Flooding0_SecondStageBN+DropOut_SecondStageLR1e-3_feature512_hidden64_DADiv16_FeatureAug',

	'input_dir':'../Cervical_cancer_data/generated_data/512x512_slide',
	'train_val_test_split_path':'../Cervical_cancer_data/generated_data/train_val_test_split_70_30_04022021.txt',
	#'train_val_test_split_path':'../../data/generated_data/tmp.txt',
	'class_path':'../Cervical_cancer_data/generated_data/class.txt',
	'out_dir':'../../data/result/Nature/',
	'n_class':4,
	'img_size':512,	
	'DADiv1':4,
	'DADiv2':4,
	'LSEPooling_scaling': 1,
	'aggregation_augmentation':True,
	'DANet':True,
	'first_stage_loss_fn':'CrossEntropyLoss',
	'fitst_stage_loss_weight':[0.02967,0.1607,0.1114,0.6983],
	'second_stage_loss_fn':'CrossEntropyLoss',
	'second_stage_loss_weight':[0.02967,0.1607,0.1114,0.6983],
	'dice_beta':0.5,

	'first_stage_feature_channel':2048,
	'second_stage_hidden_layer_num':64,
	
	'model_name':'resnet50',
	'learning_rate':1e-3,
	'learning_rate_second_stage':1e-3,
	'batch_size':8,
	'batch_size_second_stage':16,
	'num_epoch':20,
	'num_epoch_second_stage':20,
	'valid_epoch_interval':1,
	'fixed_feature':False,
	'flooding_level':0.0,

	'if_train_first_stage':True,
	'if_train_second_stage':True,
	'if_valid':True,

	'generate_heatmap':False,
	'generate_feature':False,
	
	# 0813 RESULT AMAZING
	'first_stage_pretrain_model':'08172020_Nature_2class_ResNet50_DANet_dice0.5_LR2e-4_70_30_LSEPoolScaling1_ImgAug_Flooding0_SecondStageBN+DropOut_SecondStageLR1e-3_feature512_hidden64_DADiv16',

	'mean':[138.9797284 , 149.63636513, 169.44638733],
	'std':[27.22963097, 32.56092535, 21.71684757],

	'switch_learning_rate_interval':2,
}
