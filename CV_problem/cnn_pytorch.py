"""
Pending:
	2. Fix batchnorm_3
	3. Find way to save best model
Completed:
	1. Add code for cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torchvision.transforms import transforms
from fmnist_pytorch import *
import matlplotlib.pyplot as plt
import numpy as np

def imshow(img):
	img = img * 90.13 + 82.34
	npimg = img.numpy()
	plt.imshow(npimg)
	plt.show()

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv_1 = nn.Conv2d(1, 64, 4)
		self.relu_1 = nn.ReLU()
		self.maxpool_1 = nn.MaxPool2d((2, 2))
		self.dropout_1 = nn.Dropout(p=0.1)
		self.batchnorm_1 = nn.BatchNorm2d(1)

		self.conv_2 = nn.Conv2d(64, 64, 4)
		self.relu_2 = nn.ReLU()
		self.dropout_2 = nn.Dropout(p=0.3)
		self.batchnorm_2 = nn.BatchNorm2d(64)

		self.fc_1 = nn.Linear(64*4*4, 256)
		self.dropout_3 = nn.Dropout(p=0.5)
		self.fc_2 = nn.Linear(256, 64)
		self.batchnorm_3 = nn.BatchNorm1d() 	# Fix this!
		self.fc_3 = nn.Linear(64, 4)

	def forward(self, x):
		x = self.dropout_1(self.maxpool_1(self.relu_1(self.conv_1((self.batchnorm_1(x))))))
		x = self.dropout_2(self.maxpool_2(self.relu_1(self.conv_2(x))))
		x = self.fc_3(self.batchnorm_3(self.relu_1(self.fc_2(self.dropout_3(self.relu_1(self.fc_1(flatten(x))))))))

	def flatten(self, x):
		return x.view(x.size(0), -1)

if __name__ == '__main__':

	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((82.34), (90.13))])

	trainset = fmnist('./', 28, 28, transform, 'train')
	valset = fmnist('./', 28, 28, transform, 'val')
	testset = fmnist('./', 28, 28, transform, 'test')

	trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True, num_workers=2)
	valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=4, shuffle=True, num_workers=2)
	testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=4, shuffle=True, num_workers=2)

	classes = (0, 2, 3, 6)

	dataiter = iter(trainloader)
	images, labels = dataiter.next()
	imshow(torchvision.utils.make_grid(images))
	print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

	device = torch.device("cuda:0")
	net = Net()
	net.to(device)
	max_accuracy = 0
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(10):
		running_loss = 0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs,labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			if i%50==49:
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

				correct = 0
				total = 0
				with torch.no_grad():
					for data in testloader:
						images, labels = data
						inputs, labels = inputs.to(device), labels.to(device)

						outputs = net(images)
						_, predicted = torch.max(outputs.data, 1)
						total += labels.size(0)
						correct += (predicted == labels).sum().item()

				max_accuracy = (100 * correct / total)
				print('Accuracy of the network on 2000 validation images: %d %%' % max_accuracy)

		torch.save({'epoch': epoch, 
					'model_state_dict': net.state_dict(), 
					'optimizer_state_dict': optimizer.state_dict(), 
					'loss': loss}, 'midas_%d' % epoch)