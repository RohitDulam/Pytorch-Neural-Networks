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
	noisy = img + img * noise
	return np.array(img)/ np.max(np.array(img)) #noisy/ np.max(noisy) 

def weight_init(m):
	if isinstance(m, nn.Conv2d):
		torch.nn.init.xavier_uniform_(m.weight)
		#if m.bias is not None:
			#m.bias.zero_()

class Generator(nn.Module):
	"""docstring for Generator"""
	def __init__(self):
		super(Generator, self).__init__()
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
		return F.sigmoid(self.model(inputs))

class Discriminator(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self):
		super(Discriminator, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1, padding = 1) # change input channels to 1 if MNIST, else leave it at 3.
		self.conv2 = nn.Conv2d(32, 64, 3, 1, padding = 1)
		self.conv3 = nn.Conv2d(64, 128, 3, 1, padding = 1)
		self.fc1 = nn.Linear(3 * 3 * 128, 500) ## change it to 3 * 3 * 128 if MNIST else 4 * 4 * 128 for CIFAR10. 
		self.fc2 = nn.Linear(500, 1)
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
		res3 = self.dropout(self.fc2(res2))
		return F.sigmoid(res3)

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
	params_test = {'batch_size' : batch_size, 'shuffle' : False, 'num_workers' : 0}
	training_set = data.DataLoader(train_set, **params)
	testing_set = data.DataLoader(test_set, **params_test)
	device = torch.device("cuda")
	modelG = Generator().to(device)
	modelG.apply(weight_init)
	modelD = Discriminator().to(device)
	modelD.apply(weight_init)
	criterion = nn.BCELoss().cuda()
	optimizerD = torch.optim.Adam(modelD.parameters()) # lr is the Learning Rate.
	optimizerG = torch.optim.Adam(modelG.parameters())
	i = 0

	for epoch in range(1, 101):
		train(modelG, modelD, device, criterion, optimizerG, optimizerD, training_set)
		print("Epoch number - %d" %(epoch))
		test(modelG, i)
		i += 1
		if epoch % 10 == 0:
			continue
			
		

def train(modelG, modelD, device, criterion, optimizerG, optimizerD, loader):

	modelG.train()
	modelD.train()

	for batch_idx, data in enumerate(loader):
		#print(data[0].size())
		modelD.zero_grad()
		## Loss for the Discriminator.
		label_real = Variable(torch.ones(batch_size).cuda())
		label_false = Variable(torch.zeros(batch_size).cuda())

		## Discriminator Real and Fake here itself. 
		noise = modelG(torch.randn(batch_size, 128, 4, 4).cuda()) # Change to 4, 4 if Cifar10
		noise = torch.squeeze(noise)
		input_D_R = modelD(data[0])
		input_D_F = modelD(noise)

		input_D_R = torch.squeeze(input_D_R)
		input_D_F = torch.squeeze(input_D_F)

		D_real_loss = criterion(input_D_R, label_real)
		D_real_loss.backward(retain_graph = True)

		D_fake_loss = criterion(input_D_F, label_false)
		D_fake_loss.backward(retain_graph = True)

		D_loss = D_real_loss + D_fake_loss
		optimizerD.step()

		modelG.zero_grad()

		G_loss = criterion(input_D_F, label_real)
		G_loss.backward()

		optimizerG.step()


def test(modelG, i):
	print("Testing-Phase.")
	modelG.eval()
	with torch.no_grad():
		samples = modelG(torch.rand(16, 128, 4, 4).cuda())
		fig = plot(samples)
		plt.savefig('DCGAN/{}.png'.format(str(i).zfill(3)), bbox_inches = 'tight')
		plt.close(fig)

def plot(samples):
	#print(samples.size())
	samples = torch.squeeze(samples)
	print(samples)
	#samples = samples[:16]
	figr = plt.figure(figsize = (4,4))
	gs = gridspec.GridSpec(4, 4)
	gs.update(wspace = 0.05 , hspace = 0.05)

	for i, sample in enumerate(samples):
		sample = sample.cpu().numpy()
		#sample = sample/np.max(sample)
		a = plt.subplot(gs[i])
		plt.axis('off')
		a.set_xticklabels([])
		a.set_yticklabels([])
		a.set_aspect('equal')
		plt.imshow(sample.reshape(28,28), cmap = 'gray_r')

	return figr

if not os.path.exists('DCGAN/'):
	os.makedirs('DCGAN/')

if __name__ == '__main__':
	main()



	



		
