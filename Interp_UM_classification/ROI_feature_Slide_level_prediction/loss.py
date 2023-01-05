import torch 
import torch.nn as nn

class dice_loss(nn.Module):
	def __init__(self,config):
		super().__init__()
		self.config = config
		self.beta = self.config['dice_beta']
		self.norm = nn.Softmax(dim = 1)
	def forward(self,pred,GT_labels):
		pred_norm = self.norm(pred)[:,1]
		GT_labels = GT_labels.float()
		print(GT_labels)
		print(pred_norm)
		PG = torch.sum(pred_norm*GT_labels)
		G_P = torch.sum((1-pred_norm)*GT_labels)
		P_G = torch.sum(pred_norm*(1-GT_labels))
		print("PG",PG.item())
		print("G_P",G_P.item())
		print("P_G",P_G.item())
		
		dice = (1+self.beta**2)*PG/((1+self.beta**2)*PG+self.beta**2*G_P+P_G)

		return -dice
