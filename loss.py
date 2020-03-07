import torch
import torch.nn as nn
from torch.nn import functional as F

class CE_Loss(nn.Module):
	def __init__(self):
		super(CE_Loss,self).__init__()

	def forward(self,y,x):
		S=x.shape[2]*x.shape[3]
		x=torch.log(x)
		return -torch.sum(x*y)/S

class BL_Loss(nn.Module):
	def __init__(self):
		super(BL_Loss, self).__init__()
		self.CE=CE_Loss()

	def forward(self,y1,x1,y2,x2):
		CE1=self.CE(y1,x1)
		CE2=self.CE(y2,x2)
		return CE1+CE2

class TL_Loss(nn.Module):
	def __init__(self):
		super(TL_Loss, self).__init__()
		self.CE=CE_Loss()

	def forward(self,y1,x1,y2,x2):
		CE1=self.CE(y1,x1)
		CE2=self.CE(y2,x2)
		return CE1+CE2

if __name__=='__main__':
	criterion=BL_Loss()
	input=torch.rand((1,3,320,320))
	output=torch.rand((1,3,320,320))
	loss=criterion(output,input,output,input)
	print(loss.item())