import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils import data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F

batch_size = 40

## Adding speckle to images. 

def speckle(img):
	rows, cols = img.size
	noise = np.random.rand(rows, cols)
	noise = noise.reshape(rows, cols)
	noisy = img * noise
	return np.array(img)/np.max(np.array(img)), noisy/ np.max(noisy)  

class Decoder(nn.Module):
	"""docstring for Decoder"""
	def __init__(self):
		super(Decoder, self).__init__()
		self.fc = nn.Linear(50, 4 * 4 * 128)
		self.model = nn.Sequential(
						nn.ConvTranspose2d(128, 64, 3, stride = 2, padding = 1, output_padding = 0), # nn.conv2d(in_channels, out_channels, kernel, stride, padding, padding_mode, dilation, groups, bias). Use a filter of size 4 for CIFAR with stride 2 and padding = 1 and output_padding = 0.
						nn.ReLU(),
						nn.BatchNorm2d(64),
						nn.ConvTranspose2d(64, 32, 3, stride = 2, padding = 1, output_padding = 1),
						nn.ReLU(),
						nn.BatchNorm2d(32),
						nn.ConvTranspose2d(32, 1, 3, stride = 2, padding = 1, output_padding = 1),
						nn.ReLU(),
					)

	def forward(self, inputs):
		#print("WTF", self.model(inputs).size())
		res1 = self.fc(inputs)
		#print(res1.size())
		res1 = F.relu(res1)
		res1 = res1.view(-1, 4, 4, 128)
		res1 = res1.permute(0, 3, 1, 2)
		return F.sigmoid(self.model(res1.float().cuda()))

'''class Encoder(nn.Module):
	"""docstring for Encoder"""
	def __init__(self):
		super(Encoder, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1, padding = 1) # change input channels to 1 if MNIST, else leave it at 3.
		self.conv2 = nn.Conv2d(32, 64, 3, 1, padding = 1)
		self.conv3 = nn.Conv2d(64, 128, 3, 1, padding = 1)
		self.fc1 = nn.Linear(3 * 3 * 128, 50) ## change it to 3 * 3 * 128 if MNIST else 4 * 4 * 128 for CIFAR10. 
		#self.fc2 = nn.Linear(500, 1)

	def forward(self, inputs):
		inputs = torch.unsqueeze(inputs, dim = 3).float()
		res1 = F.relu(self.conv1(inputs.permute(0, 3, 1, 2).float().cuda()))
		res1 = F.max_pool2d(res1, 2, 2)
		#print("maxpool-1",res1.size())
		res1 = F.relu(self.conv2(res1))
		res1 = F.max_pool2d(res1, 2, 2)
		#print("maxpool-2",res1.size())
		res1 = F.relu(self.conv3(res1))
		res1 = F.max_pool2d(res1, 2, 2)
		#print(res1.size())
		res2 = self.fc1(res1.view(-1, 3 * 3 * 128).cuda()) ## change it to 3 * 3 * 128 if MNIST else put 4 * 4 * 128 for CIFAR10 
		#res3 = self.fc2(res2)
		return F.sigmoid(res2)'''

class Encoder(nn.Module):
	"""docstring for Encoder"""
	def __init__(self):
		super(Encoder, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1, padding = 1) # change input channels to 1 if MNIST, else leave it at 3.
		self.conv2 = nn.Conv2d(32, 64, 3, 1, padding = 1)
		self.conv3 = nn.Conv2d(64, 128, 3, 1, padding = 1)
		self.fc1 = nn.Linear(3 * 3 * 128, 50) ## change it to 3 * 3 * 128 if MNIST else 4 * 4 * 128 for CIFAR10. 
		self.bn1 = nn.BatchNorm2d(32)
		self.bn2 = nn.BatchNorm2d(64)
		self.bn3 = nn.BatchNorm2d(128)
		self.dropout = nn.Dropout(0.5)

	def forward(self, inputs):
		inputs = torch.unsqueeze(inputs, dim = 3).float()
		res1 = self.bn1(F.relu(self.conv1(inputs.permute(0, 3, 1, 2).float().cuda())))
		res1 = F.max_pool2d(res1, 2, 2)
		#print("maxpool-1",res1.size())
		res1 = self.bn2(F.relu(self.conv2(res1)))
		res1 = F.max_pool2d(res1, 2, 2)
		#print("maxpool-2",res1.size())
		res1 = self.bn3(F.relu(self.conv3(res1)))
		res1 = F.max_pool2d(res1, 2, 2)
		#print(res1.size())
		res2 = self.dropout(self.fc1(res1.view(-1, 3 * 3 * 128).cuda())) ## change it to 3 * 3 * 128 if MNIST else put 4 * 4 * 128 for CIFAR10 
		return F.sigmoid(res2)

## Let the loss be mean square error b/w the original image and the reconstructed 

def main():

	'''train_set = datasets.CIFAR10(root='./dataCIFAR', train = True, download = True, transform = torchvision.transforms.Lambda(speckle))
	test_set = datasets.CIFAR10(root='./test_dataCIFAR', train = False, download = True, transform = torchvision.transforms.Lambda(speckle))
	params = {'batch_size' : 40, 'shuffle' : False, 'num_workers' : 0}
	params_test = {'batch_size' : 40, 'shuffle' : False, 'num_workers' : 0}
	training_set = data.DataLoader(train_set, **params)
	testing_set = data.DataLoader(test_set, **params_test)'''
	train_set = datasets.MNIST(root='./dataMNIST', train = True, download = True, transform = torchvision.transforms.Lambda(speckle))
	test_set = datasets.MNIST(root='./test_dataMNIST', train = False, download = True, transform = torchvision.transforms.Lambda(speckle))
	params = {'batch_size' : batch_size, 'shuffle' : False, 'num_workers' : 0}
	training_set = data.DataLoader(train_set, **params)
	params_test = {'batch_size' : 25, 'shuffle' : False, 'num_workers' : 0}
	testing_set = data.DataLoader(test_set, **params_test)
	device = torch.device("cuda")
	modelE = Encoder().to(device)
	modelD = Decoder().to(device)
	criterion = nn.MSELoss().cuda()
	parameters = list(modelE.parameters()) + list(modelD.parameters())
	optimizer = torch.optim.Adam(parameters, lr = 0.0001)
	i = 0
	for epoch in range(100):
		x = torch.randint(0, len(testing_set) // 25, (1,1))
		train(modelE, modelD, device, criterion, optimizer, training_set)
		print("Epoch number - %d" %(epoch))
		if epoch % 10 == 0:
			for batch_idx, data_test in enumerate(testing_set):
				if batch_idx == int(x) :
					test(modelE, modelD,device,data_test, i)
					i += 1
		

def train(modelE, modelD, device, criterion, optimizer, loader):

	modelE.train()
	modelD.train()

	for batch_idx, data_ in enumerate(loader):
		## Here there are 3 things in data. The original image.
		## Speckled Image and the label which in our case isn't useful.
		info = modelE(data_[0][1])
		output = modelD(info)
		loss = criterion(torch.squeeze(output), data_[0][0].float().cuda())
		loss.backward()
		optimizer.step()

'''def test(modelG, i):
	print("Testing-Phase.")
	modelG.eval()
	with torch.no_grad():
		samples = modelG(torch.randn(16, 128, 4, 4).cuda())
		fig = plot(samples)
		plt.savefig('finalreview/{}.png'.format(str(i).zfill(3)), bbox_inches = 'tight')
		plt.close(fig)'''

def test(modelE, modelD, device, loader, i):

	modelE.eval()
	modelD.eval()

	with torch.no_grad():
		info = modelE(loader[0][1])
		output = modelD(info)
		fig = plot(output)
		plt.savefig('EncoderDecoder/{}.png'.format(str(i).zfill(3)), bbox_inches = 'tight')
		plt.close(fig)

def plot(samples):
	samples = torch.squeeze(samples)
	#samples = samples[:16]
	figr = plt.figure(figsize = (5, 5))
	gs = gridspec.GridSpec(5 , 5)
	gs.update(wspace = 0.05 , hspace = 0.05)

	for i, sample in enumerate(samples):
		sample = sample.cpu().numpy()
		sample = sample/np.max(sample)
		a = plt.subplot(gs[i])
		plt.axis('off')
		a.set_xticklabels([])
		a.set_yticklabels([])
		a.set_aspect('equal')
		plt.imshow(sample.reshape(28,28), cmap = 'gray_r')

	return figr

if not os.path.exists('EncoderDecoder/'):
	os.makedirs('EncoderDecoder/')

if __name__ == '__main__':
	main()
