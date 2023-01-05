config = {
	'date':'06212021_Nature_2class_ResNet50_DANet_crossval_4_noAug',

	'input_dir':'../../data/generated_data/Uveal Melanoma ROI extracted',
	'train_val_test_split_path':'../../data/generated_data/Nature_cv/train_val_test_split_80_20_4_04122021.txt',
	#'train_val_test_split_path':'../../data/generated_data/tmp.txt',
	'class_path':'../../data/generated_data/class.txt',
	'out_dir':'../../data/result/Nature/',
	'n_class':2,
	'img_size':256,	
	'DADiv1':4,
	'DADiv2':4,
	'LSEPooling_scaling': 1,
	'aggregation_augmentation':False,
	'DANet':True,
	'first_stage_loss_fn':'dice_loss',
	'second_stage_loss_fn':'dice_loss',
	'dice_beta':0.5,
	'fake_num_threshold':20,

	'first_stage_feature_channel':2048,
	'second_stage_hidden_layer_num':64,
	
	'model_name':'resnet50',
	'learning_rate':2e-4,
	'learning_rate_second_stage':1e-3,
	'batch_size':24,
	'batch_size_second_stage':32,
	'num_epoch':10,
	'num_epoch_second_stage':20,
	'valid_epoch_interval':1,
	'fixed_feature':False,
	'flooding_level':0.0,

	'if_train_first_stage':True,
	'if_train_second_stage':True,
	'if_test_first_stage':False,
	'if_valid':True,
	'synthetic_validation':False,

	'generate_heatmap':False,
	'generate_feature':False,
	
	# 0813 RESULT AMAZING
	'first_stage_pretrain_model':'08172020_Nature_2class_ResNet50_DANet_dice0.5_LR2e-4_70_30_LSEPoolScaling1_ImgAug_Flooding0_SecondStageBN+DropOut_SecondStageLR1e-3_feature512_hidden64_DADiv16',
	#'first_stage_pretrain_model':'07102020_Nature_2class_ResNet50_DANet_dice0.5_LR2e-4_70_30_LSEPoolScaling1_ImgAug_Flooding0_SecondStageBN+DropOut_SecondStageLR1e-2_feature512_hidden64',
	#'first_stage_pretrain_model':'07072020_Nature_2class_ResNet50_DANet_dice0.5_LR2e-4_70_30_LSEPoolScaling1_ImgAug_Flooding0_SecondStageBN+DropOut_SecondStageLR1e-3_feature512_hidden64',
	#'first_stage_pretrain_model':'06152020_Nature_2class_ResNet50_LR1e-2_50_50_LSEPoolScaling2_ImgAug_Flooding0_SecondStageBN+DropOut_SecondStageLR0.1_feature2048_hidden64',# first model that has a reasonable heatmap
	#'first_stage_pretrain_model':'06142020_Nature_2class_ResNet50_LR1e-2_50_50_LSEPoolScaling10_ImgAug_Flooding0_SecondStageBN+DropOut_SecondStageLR0.1_feature2048_hidden64',
	#'first_stage_pretrain_model':'06092020_Nature_2class_ResNet50_50_50_LSEPool_ImgAug_Flooding0.1_SecondStageBN+DropOut_feature32_hidden8',
	#'first_stage_pretrain_model':'06072020_Nature_2class_classification_BagNet33_50_50_AvgPool_ImgAug_Flooding0.1_SecondStageBN+DropOut_feature256_hidden64',
	#'first_stage_pretrain_model':'06042020_Nature_2class_classification_BagNet33_50_50_AvgPool_ImgAug_Flooding0.1',

	'mean':[177.41, 108.29, 136.96],
	'std':[25.34, 32.26, 30.98],

	'switch_learning_rate_interval':2,
}
