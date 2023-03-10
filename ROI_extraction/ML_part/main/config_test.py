config = {
	'date': '07182020_ROI_extraction_test_pretrain_03112020_DC_wo_CTF',
	'log_dir':'../log',
	'input_dir':'../data/generated_data/CoarseExtractionV2_128',
	'out_dir':'../data/result/fine_extraction/',
	'model_name': 'bagnet17',
	'DCN_out_channel': 512,
	'cluster_vector_dim': 16,
	'fixed_feature': False,
	'n_cluster': 100,

	'resume': True,
	'pretrain_path': '../checkpoint/03112020thin6V2_bagnet17_lr-3_epoch20_NoCTF_clusterdim16_batch32_gpu2_ClusterUpdateMulti0.1_NewInitialDIRECT_NewAssignMulti0.5_UniformAss_bagnetBN0.1_KMEANInitial0.1Data_largestSTD_hierarchy.pkl',

	'data_not_train':[],
	'num_epoch':20,
	'learning_rate': 1e-3,
	'batch_size':32,
	'reassign_by_std':False,
	'M_new_assign_rand_multi':0.5,
	'cluster_update_multi': 0.1,
	'if_train':False,
	'if_valid':False,
	'valid_epoch_interval':1,
	'switch_learning_rate_interval': 3,

	'coarse_to_fine_n_cluster_step':4,
	'coarse_to_fine_epoch_step':1,
	'coarse_to_fine_epoch_start':0,
	'coarse_to_fine_epoch_max':100,
}
