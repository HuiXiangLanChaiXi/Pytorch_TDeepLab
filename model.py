import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

class backbone(nn.Module):
	def __init__(self,in_channel):
		super(backbone,self).__init__()
		self.model=torchvision.models.resnet101(pretrained=True)
		self.model.conv1=nn.Conv2d(in_channel,64,7,2,3)

	def forward(self,x):
		x=self.model.conv1(x)
		x=self.model.bn1(x)
		x=self.model.relu(x)
		x=self.model.maxpool(x)
		x=self.model.layer1(x)
		x=self.model.layer2(x)
		x=self.model.layer3(x)
		x=self.model.layer4(x)
		return x

class ASPP(nn.Module):
	def __init__(self,in_channels,out_channels,N):
		super(ASPP,self).__init__()
		self.avgpool= nn.AdaptiveAvgPool2d((1,1))
		self.conv1=nn.Sequential(
			nn.Conv2d(in_channels,out_channels//5,1,1)
			)
		self.unit1=nn.Sequential(
			nn.Conv2d(in_channels,out_channels//5,1,1),
			nn.BatchNorm2d(out_channels//5),
			nn.ReLU())
		self.unit2=nn.Sequential(
			nn.Conv2d(in_channels,out_channels//5,3,1,6,6),
			nn.BatchNorm2d(out_channels//5),
			nn.ReLU())
		self.unit3=nn.Sequential(
			nn.Conv2d(in_channels,out_channels//5,3,1,12,12),
			nn.BatchNorm2d(out_channels//5),
			nn.ReLU())
		self.unit4=nn.Sequential(
			nn.Conv2d(in_channels,out_channels//5,3,1,18,18),
			nn.BatchNorm2d(out_channels//5),
			nn.ReLU())
		self.conv2=nn.Sequential(
			nn.Conv2d(5*(out_channels//5),N,1,1)
			)
 
	def forward(self,x):
		[batch_size,C,W,H]=x.shape
		x0=self.avgpool(x)
		x0=self.conv1(x0)
		x0=F.upsample(x0,(W,H))
		x1=self.unit1(x)
		x2=self.unit2(x)
		x3=self.unit3(x)
		x4=self.unit4(x)
		x=torch.cat((x0,x1,x2,x3,x4),1)
		x=self.conv2(x)
		x=F.upsample(x,(388,388))
		return x

class TDeepLab_BL(nn.Module):
	def __init__(self):
		super(TDeepLab_BL,self).__init__()
		self.backbone=backbone(4)
		self.aspp=ASPP(2048,500,3)

	def forward(self,x1,x2):
		x=torch.cat((x1,x2),1)
		x=self.backbone(x)
		x=self.aspp(x)
		return x

class TDeepLab_TL(nn.Module):
	def __init__(self):
		super(TDeepLab_TL,self).__init__()
		self.backbone_1=backbone()
		self.backbone_2=backbone()
		self.aspp_1=ASPP(2048,500,3)
		self.aspp_2=ASPP(2048,500,3)

	def forward(self,x1,x2):

		x1=self.backbone_1(x1)
		x1=self.aspp_1(x1)

		x2=self.backbone_2(x2)
		x2=self.aspp_2(x2)

		x=(x1+x2)/2

		return x

class Siamese_TDeepLab_BL(nn.Module):
	def __init__(self):
		super(Siamese_TDeepLab_BL,self).__init__()

		self.backbone_1=backbone(4)
		self.aspp_1=ASPP(2048,500,3)

		self.backbone_2=backbone(4)
		self.aspp_2=ASPP(2048,500,3)

	def forward(self,t1_1,t1_2,t2_1,t2_2):
		t1=torch.cat((t1_1,t1_2),1)
		t2=torch.cat((t2_1,t2_2),1)

		t1=self.backbone_1(t1)
		t1=self.aspp(t1)

		t2=self.backbone_2(t2)
		t2=self.aspp(t2)

		return t1,t2

class Siamese_TDeepLab_TL(nn.Module):
	def __init__(self):
		super(Siamese_TDeepLab_TL,self).__init__()
		self.backbone1_1=backbone()
		self.backbone1_2=backbone()
		self.aspp1_1=ASPP(2048,500,3)
		self.aspp1_2=ASPP(2048,500,3)

		self.backbone2_1=backbone()
		self.backbone2_2=backbone()
		self.aspp2_1=ASPP(2048,500,3)
		self.aspp2_2=ASPP(2048,500,3)

	def forward(self,t1_1,t1_2,t2_1,t2_2):

		t1_1=self.backbone1_1(t1_1)
		t1_1=self.aspp1_1(t1_1)

		t1_2=self.backbone1_2(t1_2)
		t1_2=self.aspp1_2(t1_2)

		t1=(t1_1+t1_2)/2

		t2_1=self.backbone2_1(t2_1)
		t2_1=self.aspp2_1(t2_1)

		t2_2=self.backbone2_2(t2_2)
		t2_2=self.aspp2_2(t2_2)

		t2=(t2_1+t2_2)/2

		return x

if __name__=='__main__':
	x1=torch.rand((1,3,572,572))
	x2=torch.rand((1,1,572,572))
	model=TDeepLab_BL()
	x=model(x1,x2)
	print(x.shape)
