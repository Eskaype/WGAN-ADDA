import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function


# class disc(nn.Module):
#     def __init__(self, num_classes, ndf = 64):
#         super(disc, self).__init__()
# 		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
# 		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
# 		self.classifier = nn.Conv2d(ndf*2, 1, kernel_size=4, stride=2, padding=1)
# 		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
# 		self.up_sample = nn.Upsample(size=400, mode='bilinear')
# 		self.sigmoid = nn.Sigmoid()
#     def forward(self,x,alpha):
# 		x = self.conv1(x)
# 		x = self.leaky_relu(x)
# 		x = self.conv2(x)
# 		x = self.leaky_relu(x)
# 		x = self.classifier(x)
# 		x = self.up_sample(x)
# 		x = self.sigmoid(x)
#         return x
class ReverseLayerF(Function):

	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output=grad_output.neg()*ctx.alpha
		return output, None

class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 32):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1) #
		self.bn1 = nn.BatchNorm2d(ndf)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(ndf*2)
		#self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		#self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*2, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(size=100, mode='bilinear')
		self.up_sample = nn.Upsample(size=512, mode='bilinear')
		self.sigmoid = nn.Sigmoid()


	def forward(self, x): #, alpha):
		alpha=0.9
		x = ReverseLayerF.apply(x, alpha)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		#x = self.leaky_relu(x)
		#x = self.conv3(x)
		#x = self.leaky_relu(x)
		#x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		x = self.up_sample(x)
		x = self.sigmoid(x)
		return x

class FCDiscriminator_WGAN(nn.Module):

	def __init__(self, num_classes, ndf = 32):
		super(FCDiscriminator_WGAN, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1) #
		#self.bn1 = nn.BatchNorm2d(ndf)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		#self.bn2 = nn.BatchNorm2d(ndf*2)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		#self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*2, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(size=100, mode='bilinear')
		self.up_sample = nn.Upsample(size=512, mode='bilinear')
		self.sigmoid = nn.Sigmoid()


	def forward(self, x): #, alpha):
		#x = ReverseLayerF.apply(x, alpha)
		x = self.conv1(x)
		#x = self.bn1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		#x = self.bn2(x)
		x = self.leaky_relu(x)
		#x = self.conv3(x)
		#x = self.leaky_relu(x)
		#x = self.conv4(x)
		#x = self.leaky_relu(x)
		x = self.classifier(x)
		x = self.up_sample(x)
		clasf = x
		x = self.sigmoid(x)
		return x, clasf
