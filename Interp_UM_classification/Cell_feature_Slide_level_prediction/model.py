import torch
from torch.nn import Module, Sigmoid, Linear, ReLU, Softmax
from torch.nn.init import kaiming_uniform_, xavier_uniform_

class MLP(Module):
	
	# initialize the class
	def __init__(self, n_inputs, device):
		
		# calling constructor of parent class
		super().__init__()

		self.device = device
		
		# defining the inputs to the first hidden layer - type of hidden layer, weights, activation
		self.hid1 = Linear(n_inputs, 16) # equivalent to keras's dense layer
		kaiming_uniform_(self.hid1.weight, nonlinearity='relu') # init the weights; Common examples include the Xavier and He weight initialization schemes
		self.act1 = ReLU()
		
		# defining the inputs to the second hidden layer
		self.hid2 = Linear(16, 8)
		kaiming_uniform_(self.hid2.weight, nonlinearity='relu')
		self.act2 = ReLU()
		
		# defining the inputs to the third hidden layer (attention weights)
		self.hid_weight = Linear(8, 1)
		xavier_uniform_(self.hid_weight.weight)
		self.act3 = Sigmoid()

		# defining the inputs to the fourth hidden layer (final prediction)
		self.hid4 = Linear(8, 2)
		xavier_uniform_(self.hid_weight.weight)
		# self.act4 = Sigmoid(dim=1)


		
	def forward(self, X):
		batch_size = len(X)
		slide_preds = []

		for i in range(batch_size):
			cell_features = X[i]

			slide_feature = 0
			slide_weights_sum = 0
			# print(cell_features.shape)

			for cell_ind in range(len(cell_features)):
				cell_feature = cell_features[cell_ind][None,...]


				#input and act for layer 1
				cell_hidden_feature = self.hid1(cell_feature)
				cell_hidden_feature = self.act1(cell_hidden_feature)
				
				#input and act for layer 2
				cell_hidden_feature = self.hid2(cell_hidden_feature)
				cell_hidden_feature = self.act2(cell_hidden_feature)

				
				#weights
				weights = self.hid_weight(cell_hidden_feature)
				weights = self.act3(weights)

				slide_weights_sum += weights[0]
				slide_feature += weights[0]*cell_hidden_feature

			# slide features
			slide_feature = slide_feature / slide_weights_sum
			slide_pred = self.hid4(slide_feature)
			slide_preds.append(slide_pred[0])
		# print(slide_preds)
		slide_preds = torch.stack(slide_preds,0)
		# print(slide_preds)
		
		return slide_preds