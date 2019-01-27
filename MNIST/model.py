import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np

# Transform data to Tensor
transform = transforms.ToTensor()

# Download and load data
train_data = datasets.MNIST(root = './data', download = True, train = True, transform = transform)
test_data = datasets.MNIST(root = './data', download = True, train = False, transform = transform)

# 20% of train_data to be used for validation
valid_size = 0.2
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# Samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Data loaders
batch_size = 20
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, sampler = train_sampler)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, sampler = valid_sampler)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size)

# Define the network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # fully-connected layer(28*28 -> 512)
        self.fc1 = nn.Linear(28 * 28, 512)
        # fully-connected layer (512 -> 256)
        self.fc2 = nn.Linear(512, 256)
        # fully-connected layer (256 -> 256)
        self.fc3 = nn.Linear(256, 64)
        # fully-connected layer (64 -> 10)
        self.fc4 = nn.Linear(64, 10)
        # dropout layer (p=0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x

# initialize the neural network
model = Net()
print(model)

# Define the loss, learning rate and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

# Train the network
num_epochs = 40
min_valid_loss = np.Inf
for epoch in range(num_epochs):
	train_loss = 0.0
	valid_loss = 0.0

	model.train()
	for images, labels in train_loader:
		images = images.view(images.shape[0], -1)
		optimizer.zero_grad()
		prediction = model(images)
		loss = criterion(prediction, labels)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()*images.size(0)
	
	model.eval()
	for images, labels in valid_loader:
		prediction = model(images)
		loss = criterion(prediction, labels)
		valid_loss += loss.item()*images.size(0)

	train_loss = train_loss/len(train_loader.dataset)
	valid_loss = valid_loss/len(valid_loader.dataset)
	print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
		epoch+1, 
		train_loss,
		valid_loss
		))
	if valid_loss <= min_valid_loss:
		print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
		min_valid_loss,
		valid_loss))
		torch.save(model.state_dict(), 'trained_model.pt')
		min_valid_loss = valid_loss

model.load_state_dict(torch.load('trained_model.pt'))

test_loss = 0.0
correct_class = list(0. for i in range(10))
total_class = list(0. for i in range(10))

model.eval()

for images, labels in test_loader:
	prediction = model(images)
	loss = criterion(prediction, labels)
	test_loss += loss.item()*images.size(0)
	_, pred = torch.max(prediction, 1)
	correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
	for i in range(batch_size):
		label = labels.data[i]
		correct_class[label] += correct[i].item()
		total_class[label] += 1

test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if total_class[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * correct_class[i] / total_class[i],
            np.sum(correct_class[i]), np.sum(total_class[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(correct_class) / np.sum(total_class),
    np.sum(correct_class), np.sum(total_class)))
