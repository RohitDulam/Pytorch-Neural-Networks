import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import os
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets

def speckle(img):
	rows, cols = img.size
	noise = np.random.rand(rows, cols)
	noise = noise.reshape(rows, cols)
	noisy = img * noise
	return noisy#/ np.max(noisy) # This is for MNIST Dataset.

'''def speckle(img): ## This is for the CIFAR-10 dataset.
    rows, cols = img.size
    noise = np.random.rand(rows, cols, 3)
    noise = noise.reshape(rows, cols, 3)
    noisy = img + img * noise
    return np.array(img)#noisy/ np.max(noisy) #'''

class CNN(nn.Module):
	"""docstring for CNN"""
	def __init__(self):
		super(CNN, self).__init__()
		'''self.convmodel = nn.Sequential(
						nn.Conv2d(1, 32, 3, 1), 
						nn.ReLU(True),
						nn.MaxPool2d(2, stride = 2),
						nn.BatchNorm2d(32),
						nn.Conv2d(32, 64, 3, 1),
						nn.ReLU(True),
						nn.MaxPool2d(2, stride = 2),
						nn.BatchNorm2d(64),
						nn.Conv2d(64, 128, 3, 1),
						nn.ReLU(True),
						nn.MaxPool2d(2, stride = 2),
						nn.BatchNorm2d(128),
					)'''
		self.conv1 = nn.Conv2d(1, 32, 3, 1, padding = 1) # change input channels to 1 if MNIST, else leave it at 3.
		self.conv2 = nn.Conv2d(32, 64, 3, 1, padding = 1)
		self.conv3 = nn.Conv2d(64, 128, 3, 1, padding = 1)
		self.fc1 = nn.Linear(3 * 3 * 128, 500) ## change it to 3 * 3 * 128 if MNIST else leave it. 
		self.fc2 = nn.Linear(500, 10)

	def forward(self, inputs):
		inputs = torch.unsqueeze(inputs, dim = 3).float() # Comment this line if CIFAR10
		res1 = F.relu(self.conv1(inputs.permute(0, 3, 1, 2).float().cuda()))
		res1 = F.max_pool2d(res1, 2, 2)
		#print("maxpool-1",res1.size())
		res1 = F.relu(self.conv2(res1))
		res1 = F.max_pool2d(res1, 2, 2)
		#print("maxpool-2",res1.size())
		res1 = F.relu(self.conv3(res1))
		res1 = F.max_pool2d(res1, 2, 2)
		#print(res1.size())
		res2 = self.fc1(res1.view(-1, 3 * 3 * 128).cuda()) ## change it to 3 * 3 * 128 if MNIST else put at 4 * 4 * 128. 
		res3 = self.fc2(res2)
		return F.log_softmax(res3, dim = 1)

def train(model, device, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % 15000 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

def main():

	#train_set = datasets.CIFAR10(root='./dataCIFAR', train = True, download = True, transform = torchvision.transforms.Lambda(speckle))
	#test_set = datasets.CIFAR10(root='./test_dataCIFAR', train = False, download = True, transform = torchvision.transforms.Lambda(speckle))
	train_set = datasets.MNIST(root='./data', train = True, download = True, transform = torchvision.transforms.Lambda(speckle))
	test_set = datasets.MNIST(root='./test_data', train = False, download = True, transform = torchvision.transforms.Lambda(speckle))
	params = {'batch_size' : 40, 'shuffle' : False, 'num_workers' : 0}
	params_test = {'batch_size' : 40, 'shuffle' : False, 'num_workers' : 0}
	training_set = data.DataLoader(train_set, **params)
	testing_set = data.DataLoader(test_set, **params_test)
	device = torch.device("cuda")
	model = CNN().to(device)
	optimizer = optim.Adam(model.parameters())

	for epoch in range(1, 9):
		train(model, device,training_set, optimizer, epoch)
		test(model, device, testing_set)


if __name__ == '__main__':
	main()

		